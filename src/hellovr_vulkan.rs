use std::collections::VecDeque;
use std::ffi::{c_void, CStr, CString};
use std::fs;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::mem;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str;
use std::sync::atomic;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
use ash::vk::Handle;
//use cgmath::{Deg, Matrix4, Point3, Vector2, Vector3};
use byteorder::ReadBytesExt;
use cstr::cstr;
use failure::{err_msg, format_err, Error};
use image::io::Reader as ImageReader;
use nalgebra::base::{Matrix4, Vector2, Vector3, Vector4};
use openvr;
use openvr_sys;
use path_slash::PathExt;
use safe_transmute::transmute_to_bytes;
use unsafe_send_sync::UnsafeSend;
// XXX We can't use SDL on Windows due to bindgen issues, so we use winit instead.
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::desktop::EventLoopExtDesktop;
use winit::window::Window;

// XXX: MacOS and Xlib support are completely untested.
#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::extensions::khr::XlibSurface;
#[cfg(target_os = "macos")]
use ash::extensions::mvk::MacOSSurface;

#[cfg(target_os = "macos")]
use cocoa::appkit::{NSView, NSWindow};
#[cfg(target_os = "macos")]
use cocoa::base::id as cocoa_id;
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;

// Pipeline state objects
const PSO_SCENE: usize = 0;
const PSO_AXES: usize = 1;
const PSO_RENDERMODEL: usize = 2;
const PSO_COMPANION: usize = 3;
const PSO_COUNT: usize = 4;

const DESCRIPTOR_SET_LEFT_EYE_SCENE: usize = 0;
const DESCRIPTOR_SET_RIGHT_EYE_SCENE: usize = DESCRIPTOR_SET_LEFT_EYE_SCENE + 1;
const DESCRIPTOR_SET_COMPANION_LEFT_TEXTURE: usize = DESCRIPTOR_SET_RIGHT_EYE_SCENE + 1;
const DESCRIPTOR_SET_COMPANION_RIGHT_TEXTURE: usize = DESCRIPTOR_SET_COMPANION_LEFT_TEXTURE + 1;
const DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL0: usize = DESCRIPTOR_SET_COMPANION_RIGHT_TEXTURE + 1;
const DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL_MAX: usize =
    DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL0 + openvr::MAX_TRACKED_DEVICE_COUNT;
const DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL0: usize = DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL_MAX + 1;
const DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL_MAX: usize =
    DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL0 + openvr::MAX_TRACKED_DEVICE_COUNT;
const NUM_DESCRIPTOR_SETS: usize = DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL_MAX + 1;

struct VulkanRenderModel {
    device: Option<ash::Device>,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    image_staging_buffer: vk::Buffer,
    image_staging_buffer_memory: vk::DeviceMemory,
    constant_buffer: [vk::Buffer; 2],
    constant_buffer_memory: [vk::DeviceMemory; 2],
    // XXX: void* replaced with typed array
    constant_buffer_data: [*mut Matrix4<f32>; 2],
    descriptor_sets: [vk::DescriptorSet; 2],
    sampler: vk::Sampler,
    vertex_count: usize,
    tracked_device_index: openvr::TrackedDeviceIndex,
    _model_name: String,
}

struct MainApplication {
    // XXX: Moved from globals to fields
    vk_create_debug_report_callback_ext: Option<ash::vk::PFN_vkCreateDebugReportCallbackEXT>,
    vk_destroy_debug_report_callback_ext: Option<ash::vk::PFN_vkDestroyDebugReportCallbackEXT>,
    debug_vulkan: bool,
    verbose: bool,
    _perf: bool,
    vblank: bool,
    msaa_sample_count: u32,

    // Optional scaling factor to render with supersampling (defaults off, use -scale)
    super_sample_scale: f32,
    // XXX: Context added, instead of using a global context
    ovr_context: Option<openvr::Context>,
    hmd: Option<openvr::System>,
    // XXX: Compositor added, instead of re-querying for it
    compositor: Option<openvr::Compositor>,
    render_models: Option<openvr::RenderModels>,
    driver: String,
    display: String,
    tracked_device_pose: [openvr::TrackedDevicePose; openvr::MAX_TRACKED_DEVICE_COUNT],
    device_pose: [Matrix4<f32>; openvr::MAX_TRACKED_DEVICE_COUNT],
    show_tracked_device: [bool; openvr::MAX_TRACKED_DEVICE_COUNT],

    // XXX: SDL has been replaced by winit
    event_loop: Option<EventLoop<()>>,
    companion_window: Option<Window>,
    companion_window_width: u32,
    companion_window_height: u32,

    tracked_controller_count: i32,
    tracked_controller_count_last: i32,
    valid_pose_count: i32,
    valid_pose_count_last: i32,
    show_cubes: bool,

    // what classes we saw poses for this frame
    pose_classes: String,
    // for each device, a character representing its class
    dev_class_char: [char; openvr::MAX_TRACKED_DEVICE_COUNT],

    scene_volume_width: i32,
    scene_volume_height: i32,
    scene_volume_depth: i32,
    scale_spacing: f32,
    scale: f32,

    // if you want something other than the default 20x20x20
    scene_volume_init: i32,

    near_clip: f32,
    far_clip: f32,

    vert_count: u32,
    companion_window_index_size: u32,

    entry: Option<ash::Entry>,
    instance: Option<ash::Instance>,
    device: Option<ash::Device>,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    surface: vk::SurfaceKHR,
    surface_loader: Option<ash::extensions::khr::Surface>,
    swapchain: vk::SwapchainKHR,
    physical_device_properties: vk::PhysicalDeviceProperties,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    physical_device_features: vk::PhysicalDeviceFeatures,
    queue_family_index: u32,
    debug_report_callback: vk::DebugReportCallbackEXT,
    swap_queue_image_count: u32,
    frame_index: usize,
    current_swapchain_image: u32,
    swapchain_loader: Option<ash::extensions::khr::Swapchain>,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    swapchain_semaphores: Vec<vk::Semaphore>,
    swapchain_render_pass: vk::RenderPass,

    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; NUM_DESCRIPTOR_SETS],
    command_buffers: VecDeque<VulkanCommandBuffer>,
    current_command_buffer: VulkanCommandBuffer,

    scene_vertex_buffer: vk::Buffer,
    scene_vertex_buffer_memory: vk::DeviceMemory,
    _scene_vertex_buffer_view: vk::BufferView,
    scene_constant_buffer: [vk::Buffer; 2],
    scene_constant_buffer_memory: [vk::DeviceMemory; 2],
    // XXX: void* replaced with typed array
    scene_constant_buffer_data: [*mut Matrix4<f32>; 2],
    scene_image: vk::Image,
    scene_image_memory: vk::DeviceMemory,
    scene_image_view: vk::ImageView,
    scene_staging_buffer: vk::Buffer,
    scene_staging_buffer_memory: vk::DeviceMemory,
    scene_sampler: vk::Sampler,

    // Storage for VS and PS for each PSO
    shader_modules: [vk::ShaderModule; PSO_COUNT * 2],
    pipelines: [vk::Pipeline; PSO_COUNT],
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline_cache: vk::PipelineCache,

    // Companion window resources
    companion_window_vertex_buffer: vk::Buffer,
    companion_window_vertex_buffer_memory: vk::DeviceMemory,
    companion_window_index_buffer: vk::Buffer,
    companion_window_index_buffer_memory: vk::DeviceMemory,

    // Controller axes resources
    controller_axes_vertex_buffer: vk::Buffer,
    controller_axes_vertex_buffer_memory: vk::DeviceMemory,

    controller_vertcount: u32,

    hmd_pose: Matrix4<f32>,
    eye_pos_left: Matrix4<f32>,
    eye_pos_right: Matrix4<f32>,

    _projection_center: Matrix4<f32>,
    projection_left: Matrix4<f32>,
    projection_right: Matrix4<f32>,

    left_eye_desc: FramebufferDesc,
    right_eye_desc: FramebufferDesc,

    render_width: u32,
    render_height: u32,

    // XXX: Inline structs moved below
    render_models_vec: Vec<VulkanRenderModel>,
    tracked_device_to_render_model: [Option<usize>; openvr::MAX_TRACKED_DEVICE_COUNT],
}

#[derive(Clone, Default)]
struct VulkanCommandBuffer {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

#[repr(C)]
struct VertexDataScene {
    position: Vector3<f32>,
    tex_coord: Vector2<f32>,
}

#[repr(C)]
struct VertexDataWindow {
    position: Vector2<f32>,
    tex_coord: Vector2<f32>,
}

impl VertexDataWindow {
    fn new(position: Vector2<f32>, tex_coord: Vector2<f32>) -> Self {
        VertexDataWindow {
            position,
            tex_coord,
        }
    }
}

#[derive(Default)]
struct FramebufferDesc {
    image: vk::Image,
    image_layout: vk::ImageLayout,
    device_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    depth_stencil_image: vk::Image,
    depth_stencil_image_layout: vk::ImageLayout,
    depth_stencil_device_memory: vk::DeviceMemory,
    depth_stencil_image_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
}

static GLOBAL_SHOULD_PRINT: atomic::AtomicBool = atomic::AtomicBool::new(true);

//-----------------------------------------------------------------------------
// Purpose: Outputs a set of optional arguments to debugging output, using
//          the printf format setting specified in fmt*.
//-----------------------------------------------------------------------------
macro_rules! dprintln {
    ($fmt:expr) => {if GLOBAL_SHOULD_PRINT.load(atomic::Ordering::SeqCst) { eprintln!($fmt); } };
    ($fmt:expr, $($arg:tt)*) => {if GLOBAL_SHOULD_PRINT.load(atomic::Ordering::SeqCst) { eprintln!($fmt, $($arg)*); } };
}

// XXX: This allows getting a backtrace for validation errors, which is useful for debugging.
static GLOBAL_ERRORS_SHOULD_PANIC: atomic::AtomicBool = atomic::AtomicBool::new(false);

//-----------------------------------------------------------------------------
// Purpose: VK_EXT_debug_report callback
//-----------------------------------------------------------------------------
#[no_mangle]
unsafe extern "system" fn debug_message_callback(
    flags: vk::DebugReportFlagsEXT,
    _object_type: vk::DebugReportObjectTypeEXT,
    _object: u64,
    location: usize,
    message_code: i32,
    layer_prefix: *const c_char,
    message: *const c_char,
    _userdata: *mut c_void,
) -> vk::Bool32 {
    let layer_prefix = CStr::from_ptr(layer_prefix).to_str().unwrap();
    let message = CStr::from_ptr(message).to_str().unwrap();
    if flags.contains(vk::DebugReportFlagsEXT::ERROR) {
        let msg = format!(
            "VK ERROR {} {} :{}: {}",
            layer_prefix, location, message_code, message
        );
        dprintln!("{}", &msg);
        if GLOBAL_ERRORS_SHOULD_PANIC.load(atomic::Ordering::SeqCst) {
            panic!(msg);
        }
    } else if flags.contains(vk::DebugReportFlagsEXT::WARNING) {
        dprintln!(
            "VK WARNING {} {} :{}: {}",
            layer_prefix,
            location,
            message_code,
            message
        );
    } else if flags.contains(vk::DebugReportFlagsEXT::PERFORMANCE_WARNING) {
        dprintln!(
            "VK PERF {} {} :{}: {}",
            layer_prefix,
            location,
            message_code,
            message
        );
    } else if flags.contains(vk::DebugReportFlagsEXT::INFORMATION) {
        dprintln!(
            "VK INFO {} {} :{}: {}",
            layer_prefix,
            location,
            message_code,
            message
        );
    } else if flags.contains(vk::DebugReportFlagsEXT::DEBUG) {
        dprintln!(
            "VK DEBUG {} {} :{}: {}",
            layer_prefix,
            location,
            message_code,
            message
        );
    }
    return 0;
}

//-----------------------------------------------------------------------------
// Purpose: Determine the memory type index from the memory requirements
// and type bits
//-----------------------------------------------------------------------------
fn memory_type_from_properties(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    mut memory_type_bits: u32,
    memory_property_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..vk::MAX_MEMORY_TYPES {
        if (memory_type_bits & 1) == 1 {
            // Type is available, does it match user properties?
            if (memory_properties.memory_types[i].property_flags & memory_property_flags)
                == memory_property_flags
            {
                return Some(i as u32);
            }
        }
        memory_type_bits >>= 1;
    }
    // No memory types matched, return failure
    return None;
}

fn create_vulkan_buffer<T>(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    buffer_data: Option<&[T]>,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), vk::Result> {
    let buffer_create_info = vk::BufferCreateInfo::builder().size(size).usage(usage);
    let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };
    let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let alloc_info = vk::MemoryAllocateInfo::builder()
        .memory_type_index(
            memory_type_from_properties(
                memory_properties,
                memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )
            .expect("Failed to find matching memory type index for buffer"),
        )
        .allocation_size(memory_requirements.size);
    let device_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe {
        device.bind_buffer_memory(buffer, device_memory, 0)?;
    }
    if let Some(buffer_data) = buffer_data {
        unsafe {
            let data = device.map_memory(
                device_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::default(),
            )?;
            data.copy_from_nonoverlapping(buffer_data.as_ptr() as *const c_void, size as usize);
            // XXX: Spec says we need to unmap *after* flushing.
        }
        let memory_range = vk::MappedMemoryRange::builder()
            .memory(device_memory)
            .size(vk::WHOLE_SIZE)
            .build();
        unsafe {
            device.flush_mapped_memory_ranges(&[memory_range])?;
            device.unmap_memory(device_memory);
        }
    }
    return Ok((buffer, device_memory));
}

impl MainApplication {
    fn new(args: impl Iterator<Item = String>) -> Self {
        let mut app = MainApplication {
            vk_create_debug_report_callback_ext: None,
            vk_destroy_debug_report_callback_ext: None,
            debug_vulkan: bool::default(),
            verbose: bool::default(),
            _perf: bool::default(),
            vblank: bool::default(),
            msaa_sample_count: u32::default(),

            super_sample_scale: f32::default(),
            ovr_context: None,
            hmd: None,
            compositor: None,
            render_models: None,
            driver: String::default(),
            display: String::default(),
            tracked_device_pose: unsafe { mem::zeroed() },
            device_pose: [Matrix4::<f32>::default(); openvr::MAX_TRACKED_DEVICE_COUNT],
            show_tracked_device: [bool::default(); openvr::MAX_TRACKED_DEVICE_COUNT],

            event_loop: None,
            companion_window: None,
            companion_window_width: u32::default(),
            companion_window_height: u32::default(),

            tracked_controller_count: i32::default(),
            tracked_controller_count_last: i32::default(),
            valid_pose_count: i32::default(),
            valid_pose_count_last: i32::default(),
            show_cubes: bool::default(),

            // what classes we saw poses for this frame
            pose_classes: String::default(),
            // for each device, a character representing its class
            dev_class_char: [char::default(); openvr::MAX_TRACKED_DEVICE_COUNT],

            scene_volume_width: i32::default(),
            scene_volume_height: i32::default(),
            scene_volume_depth: i32::default(),
            scale_spacing: f32::default(),
            scale: f32::default(),

            scene_volume_init: 20,

            near_clip: f32::default(),
            far_clip: f32::default(),

            vert_count: u32::default(),
            companion_window_index_size: u32::default(),

            entry: None,
            instance: None,
            device: None,
            physical_device: vk::PhysicalDevice::default(),
            queue: vk::Queue::default(),
            surface: vk::SurfaceKHR::default(),
            surface_loader: None,
            swapchain_loader: None,
            swapchain: vk::SwapchainKHR::default(),
            physical_device_properties: vk::PhysicalDeviceProperties::default(),
            physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties::default(),
            physical_device_features: vk::PhysicalDeviceFeatures::default(),
            queue_family_index: u32::default(),
            debug_report_callback: vk::DebugReportCallbackEXT::default(),
            swap_queue_image_count: u32::default(),
            frame_index: usize::default(),
            current_swapchain_image: u32::default(),
            swapchain_images: Vec::default(),
            swapchain_image_views: Vec::default(),
            swapchain_framebuffers: Vec::default(),
            swapchain_semaphores: Vec::default(),
            swapchain_render_pass: vk::RenderPass::default(),

            command_pool: vk::CommandPool::default(),
            descriptor_pool: vk::DescriptorPool::default(),
            descriptor_sets: [vk::DescriptorSet::default(); NUM_DESCRIPTOR_SETS],
            command_buffers: VecDeque::new(),
            current_command_buffer: VulkanCommandBuffer::default(),

            scene_vertex_buffer: vk::Buffer::default(),
            scene_vertex_buffer_memory: vk::DeviceMemory::default(),
            _scene_vertex_buffer_view: vk::BufferView::default(),
            scene_constant_buffer: [vk::Buffer::default(); 2],
            scene_constant_buffer_memory: [vk::DeviceMemory::default(); 2],
            scene_constant_buffer_data: [ptr::null_mut(); 2],
            scene_image: vk::Image::default(),
            scene_image_memory: vk::DeviceMemory::default(),
            scene_image_view: vk::ImageView::default(),
            scene_staging_buffer: vk::Buffer::default(),
            scene_staging_buffer_memory: vk::DeviceMemory::default(),
            scene_sampler: vk::Sampler::default(),

            shader_modules: [vk::ShaderModule::default(); PSO_COUNT * 2],
            pipelines: [vk::Pipeline::default(); PSO_COUNT],
            descriptor_set_layout: vk::DescriptorSetLayout::default(),
            pipeline_layout: vk::PipelineLayout::default(),
            pipeline_cache: vk::PipelineCache::default(),

            companion_window_vertex_buffer: vk::Buffer::default(),
            companion_window_vertex_buffer_memory: vk::DeviceMemory::default(),
            companion_window_index_buffer: vk::Buffer::default(),
            companion_window_index_buffer_memory: vk::DeviceMemory::default(),

            controller_axes_vertex_buffer: vk::Buffer::default(),
            controller_axes_vertex_buffer_memory: vk::DeviceMemory::default(),

            controller_vertcount: u32::default(),

            hmd_pose: Matrix4::<f32>::default(),
            eye_pos_left: Matrix4::<f32>::default(),
            eye_pos_right: Matrix4::<f32>::default(),

            _projection_center: Matrix4::<f32>::default(),
            projection_left: Matrix4::<f32>::default(),
            projection_right: Matrix4::<f32>::default(),

            left_eye_desc: FramebufferDesc::default(),
            right_eye_desc: FramebufferDesc::default(),

            render_width: u32::default(),
            render_height: u32::default(),

            render_models_vec: Vec::default(),
            tracked_device_to_render_model: [None; openvr::MAX_TRACKED_DEVICE_COUNT],
        };
        app.companion_window_width = 640;
        app.companion_window_height = 320;
        app.msaa_sample_count = 4;
        app.super_sample_scale = 1.;
        app.scene_volume_init = 20;
        app.show_cubes = true;
        let args: Vec<String> = args.collect();
        for (i, arg) in args.iter().enumerate() {
            match arg.as_str() {
                "-Vulkandebug" => {
                    app.debug_vulkan = true;
                    // XXX: Panic on validation errors to get useful backtraces
                    GLOBAL_ERRORS_SHOULD_PANIC.store(true, atomic::Ordering::SeqCst);
                }
                "-verbose" => app.verbose = true,
                "-novblank" => app.vblank = false,
                "-msaa" if args.len() > i + 1 && !args[i + 1].starts_with('-') => {
                    app.msaa_sample_count = args[i + 1].parse().expect("Invalid msaa argument");
                }
                "-supersampler" if args.len() > i + 1 && !args[i + 1].starts_with('-') => {
                    app.super_sample_scale =
                        args[i + 1].parse().expect("Invalid supersample argument");
                }
                "-noprintf" => {
                    GLOBAL_SHOULD_PRINT.store(false, atomic::Ordering::SeqCst);
                }
                "-cubevolume" if args.len() > i + 1 && !args[i + 1].starts_with('-') => {
                    app.scene_volume_init =
                        args[i + 1].parse().expect("Invalid cubevolume argument");
                }
                _ => {}
            }
        }
        app
    }
}

//-----------------------------------------------------------------------------
// Purpose: Helper to get a string from a tracked device property and turn it
//			into a std::String
//-----------------------------------------------------------------------------
fn get_tracked_device_string(
    hmd: &openvr::System,
    device: openvr::TrackedDeviceIndex,
    prop: openvr::TrackedDeviceProperty,
) -> Result<String, openvr::system::TrackedPropertyError> {
    hmd.string_tracked_device_property(device, prop)
        .map(|c_str| c_str.to_string_lossy().into_owned())
}

impl MainApplication {
    fn init(&mut self) -> Result<(), Error> {
        // Loading the SteamVR Runtime
        unsafe {
            let ovr_context = openvr::init(openvr::ApplicationType::Scene)?;
            self.hmd = Some(ovr_context.system()?);
            self.compositor = Some(ovr_context.compositor()?);
            self.compositor
                .as_ref()
                .unwrap()
                .set_tracking_space(openvr::TrackingUniverseOrigin::Standing);
            self.render_models = Some(ovr_context.render_models()?);
            self.ovr_context = Some(ovr_context);
        }

        let window_pos_x = 700;
        let window_pos_y = 100;

        let event_loop = EventLoop::new();
        let window = Window::new(&event_loop)?;
        window.set_title("hellovr [Vulkan] [Rust]");
        window.set_outer_position(winit::dpi::PhysicalPosition {
            x: window_pos_x,
            y: window_pos_y,
        });
        window.set_inner_size(winit::dpi::PhysicalSize {
            width: self.companion_window_width,
            height: self.companion_window_height,
        });
        window.set_resizable(false);
        window.set_visible(true);
        self.event_loop = Some(event_loop);
        self.companion_window = Some(window);

        self.driver = get_tracked_device_string(
            self.hmd.as_ref().unwrap(),
            openvr::tracked_device_index::HMD,
            openvr::property::TrackingSystemName_String,
        )
        .unwrap_or_else(|_| "No Driver".to_owned());
        self.display = get_tracked_device_string(
            self.hmd.as_ref().unwrap(),
            openvr::tracked_device_index::HMD,
            openvr::property::SerialNumber_String,
        )
        .unwrap_or_else(|_| "No Display".to_owned());

        self.companion_window.as_mut().unwrap().set_title(&format!(
            "hellovr [Vulkan] [Rust] - {} {}",
            self.driver, self.display
        ));

        // cube array
        self.scene_volume_width = self.scene_volume_init;
        self.scene_volume_height = self.scene_volume_init;
        self.scene_volume_depth = self.scene_volume_init;

        self.scale = 0.3;
        self.scale_spacing = 1.0;

        self.near_clip = 0.1;
        self.far_clip = 300.0;

        // XXX: These are already initialized
        self.vert_count = 0;
        self.companion_window_index_size = 0;

        self.init_vulkan()?;

        self.init_compositor()?;

        Ok(())
    }

    //--------------------------------------------------------------------------------------
    // Ask OpenVR for the list of instance extensions required
    //--------------------------------------------------------------------------------------
    fn get_vulkan_instance_extensions_required(&self) -> Option<Vec<CString>> {
        Some(
            self.compositor
                .as_ref()?
                .vulkan_instance_extensions_required(),
        )
    }

    //--------------------------------------------------------------------------------------
    // Ask OpenVR for the list of device extensions required
    //--------------------------------------------------------------------------------------
    fn get_vulkan_device_extensions_required(&self) -> Option<Vec<CString>> {
        let physical_device = self.physical_device.as_raw() as *mut openvr::VkPhysicalDevice_T;
        unsafe {
            Some(
                self.compositor
                    .as_ref()?
                    .vulkan_device_extensions_required(physical_device),
            )
        }
    }

    //-----------------------------------------------------------------------------
    // Purpose: Initialize Vulkan VkInstance
    //-----------------------------------------------------------------------------
    fn init_vulkan_instance(&mut self) -> Result<(), Error> {
        let entry = ash::Entry::new()?;
        let mut required_instance_extensions = self
            .get_vulkan_instance_extensions_required()
            .ok_or_else(|| err_msg("Could not determine OpenVR Vulkan instance extensions."))?;
        required_instance_extensions.push(ash::extensions::khr::Surface::name().to_owned());
        // XXX: These instance extensions aren't listed as required by OpenVR, but are required by
        // device extensions listed as required by OpenVR.
        required_instance_extensions
            .push(CString::new("VK_KHR_external_semaphore_capabilities").unwrap());
        required_instance_extensions
            .push(CString::new("VK_KHR_external_fence_capabilities").unwrap());
        if cfg!(windows) {
            required_instance_extensions
                .push(ash::extensions::khr::Win32Surface::name().to_owned());
        } else if cfg!(macos) {
            // XXX: Try to support macos, I guess
            required_instance_extensions
                .push(ash::extensions::mvk::MacOSSurface::name().to_owned());
        } else {
            required_instance_extensions.push(ash::extensions::khr::XlibSurface::name().to_owned());
        }
        let mut enabled_layer_names = Vec::new();

        // Enable validation layers
        if self.debug_vulkan {
            // OpenVR: no unique_objects when using validation with SteamVR
            let instance_validation_layers = [
                "VK_LAYER_GOOGLE_threading",
                "VK_LAYER_LUNARG_parameter_validation",
                "VK_LAYER_LUNARG_object_tracker",
                "VK_LAYER_LUNARG_image",
                "VK_LAYER_LUNARG_core_validation",
                "VK_LAYER_LUNARG_swapchain",
                // XXX: Add KHRONOS validation layer
                "VK_LAYER_KHRONOS_validation",
            ];

            let instance_layer_properties = entry.enumerate_instance_layer_properties()?;
            for property in instance_layer_properties {
                if let Ok(name) = vulkan_str(&property.layer_name) {
                    for layer in &instance_validation_layers {
                        // XXX: This also enables layers with names that start with the above
                        if name.find(layer).is_some() {
                            enabled_layer_names.push(CString::new(name).unwrap());
                        }
                    }
                }
            }
            if enabled_layer_names.len() > 0 {
                required_instance_extensions
                    .push(ash::extensions::ext::DebugReport::name().to_owned());
            }
        }
        let instance_extension_properties = entry.enumerate_instance_extension_properties()?;
        let mut enabled_instance_extension_names =
            Vec::with_capacity(required_instance_extensions.len());
        for required_extension in &required_instance_extensions {
            let mut found = false;
            for extension in &instance_extension_properties {
                if vulkan_str(&extension.extension_name) == required_extension.to_str() {
                    found = true;
                    enabled_instance_extension_names.push(required_extension.clone());
                    break;
                }
            }
            if !found {
                dprintln!(
                    "Vulkan missing requested extension {:?}.",
                    required_extension
                );
            }
        }
        if enabled_instance_extension_names.len() != required_instance_extensions.len() {
            return Err(err_msg(
                "Could not get required Vulkan instance extensions.",
            ));
        }

        self.instance = {
            let app_info = vk::ApplicationInfo::builder()
                .application_name(cstr!("hellovr_vulkan_rust"))
                .application_version(1)
                .engine_version(1)
                .api_version(ash::vk_make_version!(1, 0, 0));
            // XXX: These Vec's must live until the end of the unsafe block below.
            let raw_extension_names = to_raw_cstr_array(&enabled_instance_extension_names);
            let raw_layer_names = to_raw_cstr_array(&enabled_layer_names);
            let instance_create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&raw_extension_names)
                .enabled_layer_names(&raw_layer_names);

            Some(unsafe { entry.create_instance(&instance_create_info, None) }?)
        };

        if self.debug_vulkan {
            let vk_create_debug_report_callback_ext: ash::vk::PFN_vkCreateDebugReportCallbackEXT = unsafe {
                mem::transmute(
                    entry
                        .get_instance_proc_addr(
                            self.instance.as_ref().unwrap().handle(),
                            mem::transmute(b"vkCreateDebugReportCallbackEXT\0" as *const u8),
                            //cstr!("vkCreateDebugReportCallbackEXT").as_ptr(),
                        )
                        .ok_or_else(|| {
                            err_msg("Could not lookup vkCreateDebugReportCallbackEXT")
                        })?,
                )
            };
            self.vk_create_debug_report_callback_ext = Some(vk_create_debug_report_callback_ext);
            let vk_destroy_debug_report_callback_ext: ash::vk::PFN_vkDestroyDebugReportCallbackEXT = unsafe {
                mem::transmute(
                    entry
                        .get_instance_proc_addr(
                            self.instance.as_ref().unwrap().handle(),
                            cstr!("vkDestroyDebugReportCallbackEXT").as_ptr(),
                        )
                        .ok_or_else(|| {
                            err_msg("Could not lookup vkDestroyDebugReportCallbackEXT")
                        })?,
                )
            };
            self.vk_destroy_debug_report_callback_ext = Some(vk_destroy_debug_report_callback_ext);
            let debug_report_create_info = vk::DebugReportCallbackCreateInfoEXT::builder()
                .flags(vk::DebugReportFlagsEXT::ERROR)
                .pfn_callback(Some(debug_message_callback));
            vk_create_debug_report_callback_ext(
                self.instance.as_ref().unwrap().handle(),
                &debug_report_create_info.build(),
                ptr::null(),
                &mut self.debug_report_callback,
            );
        }
        self.entry = Some(entry);

        Ok(())
    }

    fn init_vulkan_device(&mut self) -> Result<(), Error> {
        let instance = self.instance.as_ref().unwrap();

        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let instance_handle = instance.handle().as_raw() as *mut openvr::VkInstance_T;
        // Query OpenVR for the physical device to use
        if let Some(device_handle) = self
            .hmd
            .as_ref()
            .and_then(|hmd| hmd.vulkan_output_device(instance_handle))
        {
            // Select the HMD physical device
            for device in &physical_devices {
                if device_handle as u64 == device.as_raw() {
                    self.physical_device = device.clone();
                }
            }
        }
        if self.physical_device == vk::PhysicalDevice::null() {
            // Fallback: Grab the first physical device
            dprintln!("Failed to find GetOutputDevice VkPhysicalDevice, falling back to choosing first device.");
            self.physical_device = physical_devices[0];
        }

        self.physical_device_properties =
            unsafe { instance.get_physical_device_properties(self.physical_device) };
        self.physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(self.physical_device) };
        self.physical_device_features =
            unsafe { instance.get_physical_device_features(self.physical_device) };
        //--------------------//
        // VkDevice creation  //
        //--------------------//
        // Query OpenVR for the required device extensions for this physical device
        let mut required_device_extensions = self.get_vulkan_device_extensions_required().unwrap();
        // Add additional required extensions
        required_device_extensions.push(ash::extensions::khr::Swapchain::name().to_owned());

        // Find the first graphics queue
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(self.physical_device) };

        let graphics_queue_index = queue_family_properties
            .iter()
            .position(|props| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| format_err!("No graphics queue found"))?;

        self.queue_family_index = graphics_queue_index as u32;

        let device_extensions =
            unsafe { instance.enumerate_device_extension_properties(self.physical_device)? };

        let mut enabled_extension_names = Vec::with_capacity(required_device_extensions.len());

        for required_extension in &required_device_extensions {
            let mut ext_found = false;
            for available_extension in &device_extensions {
                if vulkan_str(&available_extension.extension_name) == required_extension.to_str() {
                    ext_found = true;
                    enabled_extension_names.push(required_extension.clone());
                    break;
                }
            }
            // XXX: Added this error reporting
            if !ext_found {
                dprintln!("Could not get required extension {:?}", required_extension);
            }
        }

        let raw_extension_names = to_raw_cstr_array(&enabled_extension_names);
        let device_queue_create_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(self.queue_family_index)
            .queue_priorities(&[1.])
            .build()];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&device_queue_create_infos)
            .enabled_extension_names(&raw_extension_names)
            .enabled_features(&self.physical_device_features);

        self.device = Some(unsafe {
            instance.create_device(self.physical_device, &device_create_info, None)
        }?);

        // Get the device queue
        self.queue = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(self.queue_family_index, 0)
        };
        Ok(())
    }

    //-----------------------------------------------------------------------------
    // Purpose: Initialize Vulkan swapchain and associated images
    //-----------------------------------------------------------------------------
    fn init_vulkan_swapchain(&mut self) -> Result<(), Error> {
        let entry = self.entry.as_ref().unwrap();
        let instance = self.instance.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();
        self.surface =
            unsafe { create_surface(entry, instance, self.companion_window.as_ref().unwrap()) }?;
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);
        self.surface_loader = Some(surface_loader.clone());

        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                self.physical_device,
                self.queue_family_index,
                self.surface,
            )
        };
        if !supports_present {
            return Err(err_msg(
                "vkGetPhysicalDeviceSurfaceSupportKHR present not supported.",
            ));
        }
        let supported_swapchain_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(self.physical_device, self.surface)
        }?;

        let mut swapchain_format = None;

        // Favor sRGB if it's available
        for format in &supported_swapchain_formats {
            if swapchain_format.is_none()
                && (format.format == vk::Format::B8G8R8A8_SRGB
                    || format.format == vk::Format::R8G8B8A8_SRGB)
            {
                swapchain_format = Some(format.clone());
            }
        }
        // Default to the first one if no sRGB
        let swapchain_format = swapchain_format.unwrap_or(supported_swapchain_formats[0].clone());
        let color_space = swapchain_format.color_space;

        // Check the surface properties and formats
        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }?;

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
        }?;

        // width and height are either both -1, or both not -1.
        // XXX: The above comment says -1, but extents are composed of u32
        let swapchain_extent = if surface_caps.current_extent.width == u32::MAX {
            // If the surface size is undefined, the size is set to the size of the images requested.
            vk::Extent2D {
                width: self.companion_window_width,
                height: self.companion_window_height,
            }
        } else {
            surface_caps.current_extent
        };

        // Order of preference for no vsync:
        // 1. VK_PRESENT_MODE_IMMEDIATE_KHR - The presentation engine does not wait for a vertical blanking period to update the current image,
        //                                    meaning this mode may result in visible tearing
        // 2. VK_PRESENT_MODE_MAILBOX_KHR - The presentation engine waits for the next vertical blanking period to update the current image. Tearing cannot be observed.
        //                                  An internal single-entry queue is used to hold pending presentation requests.
        // 3. VK_PRESENT_MODE_FIFO_RELAXED_KHR - equivalent of eglSwapInterval(-1).
        let swapchain_present_mode = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO_RELAXED,
        ]
        .iter()
        .cloned()
        .find(|mode| present_modes.contains(mode))
        .unwrap_or(vk::PresentModeKHR::FIFO);

        // 4. VK_PRESENT_MODE_FIFO_KHR - equivalent of eglSwapInterval(1).  The presentation engine
        // waits for the next vertical blanking period to update the current image. Tearing cannot
        // be observed. This mode must be supported by all implementations.

        // Have a swap queue depth of at least three frames
        // XXX I don't know how to interpret the above comment. It seems wrong.
        self.swap_queue_image_count = u32::max(surface_caps.min_image_count, 2);
        if surface_caps.max_image_count > 0
            && self.swap_queue_image_count > surface_caps.max_image_count
        {
            // Application must settle for fewer images than desired:
            self.swap_queue_image_count = surface_caps.max_image_count;
        }

        let pre_transform = if surface_caps
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_caps.current_transform
        };

        let mut image_usage_flags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if surface_caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            image_usage_flags |= vk::ImageUsageFlags::TRANSFER_DST;
        } else {
            dprintln!("Vulkan swapchain does not support VK_IMAGE_USAGE_TRANSFER_DST_BIT. Some operations may not be supported.");
        }

        let composite_alpha = if surface_caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else if surface_caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::INHERIT)
        {
            vk::CompositeAlphaFlagsKHR::INHERIT
        } else {
            dprintln!(
                "Unexpected value for VkSurfaceCapabilitiesKHR.compositeAlpha: {:?}\n",
                surface_caps.supported_composite_alpha
            );
            surface_caps.supported_composite_alpha
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(self.swap_queue_image_count)
            .image_format(swapchain_format.format)
            .image_color_space(color_space)
            .image_extent(swapchain_extent)
            .image_usage(image_usage_flags)
            .pre_transform(pre_transform)
            .image_array_layers(1)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .present_mode(swapchain_present_mode)
            .clipped(true)
            .composite_alpha(composite_alpha);

        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
        self.swapchain =
            unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }?;

        self.swapchain_images =
            unsafe { swapchain_loader.get_swapchain_images(self.swapchain.clone()) }?;
        self.swapchain_loader = Some(swapchain_loader);

        // Create a renderpass
        let attachment_references = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let attachment_desc = [vk::AttachmentDescription {
            format: swapchain_format.format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            flags: vk::AttachmentDescriptionFlags::empty(),
        }];

        let sub_pass_create_info = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_references)
            .build()];

        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_desc)
            .subpasses(&sub_pass_create_info);

        self.swapchain_render_pass =
            unsafe { device.create_render_pass(&render_pass_create_info, None) }?;

        for image in &self.swapchain_images {
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image.clone())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let image_view = unsafe { device.create_image_view(&image_view_create_info, None) }?;
            self.swapchain_image_views.push(image_view);
            let attachments = [image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(self.swapchain_render_pass)
                .attachments(&attachments)
                .width(self.companion_window_width)
                .height(self.companion_window_height)
                .layers(1);
            let framebuffer = unsafe { device.create_framebuffer(&framebuffer_create_info, None) }?;
            self.swapchain_framebuffers.push(framebuffer);
            let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
            self.swapchain_semaphores
                .push(unsafe { device.create_semaphore(&semaphore_create_info, None) }?);
        }
        Ok(())
    }

    //-----------------------------------------------------------------------------
    // Purpose: Initialize Vulkan. Returns true if Vulkan has been successfully
    //          initialized, false if shaders could not be created.
    //          If failure occurred in a module other than shaders, the function
    //          may return true or throw an error.
    //-----------------------------------------------------------------------------
    fn init_vulkan(&mut self) -> Result<(), Error> {
        self.init_vulkan_instance()?;
        self.init_vulkan_device()?;
        self.init_vulkan_swapchain()?;

        // Create the command pool
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(self.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        self.command_pool = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_command_pool(&command_pool_create_info, None)
        }?;

        // Command buffer used during resource loading
        self.current_command_buffer = self.get_command_buffer()?;
        {
            let device = self.device.as_ref().unwrap();

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                device.begin_command_buffer(
                    self.current_command_buffer.command_buffer.clone(),
                    &command_buffer_begin_info,
                )
            }?;
        }

        self.setup_texture_maps()?;
        self.setup_scene()?;
        self.setup_cameras();
        self.setup_stereo_render_targets()?;
        self.setup_companion_window()?;

        self.create_all_shaders()?;

        self.create_all_descriptor_sets()?;
        self.setup_render_models()?;

        // Submit the command buffer used during loading
        {
            let device = self.device.as_ref().unwrap();
            unsafe {
                device.end_command_buffer(self.current_command_buffer.command_buffer.clone())
            }?;
            let command_buffers = [self.current_command_buffer.command_buffer.clone()];
            let submit_info = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers[..])
                .build()];
            unsafe {
                device
                    .queue_submit(self.queue, &submit_info, self.current_command_buffer.fence)
                    .unwrap()
            };
            self.current_command_buffer.command_buffer = vk::CommandBuffer::null();
            self.current_command_buffer.fence = vk::Fence::null();

            unsafe { device.queue_wait_idle(self.queue) }?;
        }

        Ok(())
    }

    //-----------------------------------------------------------------------------
    // Purpose: Initialize Compositor. Returns true if the compositor was
    //          successfully initialized, false otherwise.
    //-----------------------------------------------------------------------------
    fn init_compositor(&mut self) -> Result<(), Error> {
        self.compositor = Some(self.ovr_context.as_ref().unwrap().compositor()?);
        Ok(())
    }

    //-----------------------------------------------------------------------------
    // Purpose: Get an available command buffer or create a new one if
    // none available.  Associate a fence with the command buffer.
    //-----------------------------------------------------------------------------
    fn get_command_buffer(&mut self) -> Result<VulkanCommandBuffer, Error> {
        // If the fence associated with the command buffer has finished, reset it and return it
        let device = self.device.as_ref().unwrap();
        if self.command_buffers.len() > 0 {
            let status = unsafe {
                device.get_fence_status(
                    self.command_buffers
                        .back()
                        .expect("Already checked length")
                        .fence,
                )
            };
            if status.is_ok() || status == Err(vk::Result::SUCCESS) {
                let cmd_buffer = self
                    .command_buffers
                    .pop_back()
                    .expect("Already checked length");
                unsafe {
                    device.reset_command_buffer(
                        cmd_buffer.command_buffer.clone(),
                        vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                    )?;
                    device.reset_fences(&[cmd_buffer.fence])?;
                }
                return Ok(cmd_buffer);
            }
        }
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(self.command_pool.clone())
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffer =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }?[0];
        let fence_create_info = vk::FenceCreateInfo::builder();
        let fence = unsafe { device.create_fence(&fence_create_info, None) }?;
        return Ok(VulkanCommandBuffer {
            command_buffer,
            fence,
        });
    }

    fn shutdown(&mut self) -> Result<(), Error> {
        dprintln!("Shutdown");
        if let Some(device) = self.device.as_ref() {
            unsafe { device.device_wait_idle() }?;
        }
        if let Some(context) = self.ovr_context.as_ref() {
            unsafe {
                context.shutdown();
            }
        }
        self.render_models_vec.clear();
        if let Some(device) = self.device.as_ref() {
            unsafe {
                while let Some(cmd_buf) = self.command_buffers.pop_back() {
                    device.free_command_buffers(self.command_pool, &[cmd_buf.command_buffer]);
                    device.destroy_fence(cmd_buf.fence, None);
                }
                device.destroy_command_pool(self.command_pool, None);
                device.destroy_descriptor_pool(self.descriptor_pool, None);

                for framebuffer_desc in &[&self.left_eye_desc, &self.right_eye_desc] {
                    if framebuffer_desc.image_view != vk::ImageView::null() {
                        device.destroy_image_view(framebuffer_desc.image_view, None);
                        device.destroy_image(framebuffer_desc.image, None);
                        device.free_memory(framebuffer_desc.device_memory, None);
                        device.destroy_image_view(framebuffer_desc.depth_stencil_image_view, None);
                        device.destroy_image(framebuffer_desc.depth_stencil_image, None);
                        device.free_memory(framebuffer_desc.depth_stencil_device_memory, None);
                        device.destroy_render_pass(framebuffer_desc.render_pass, None);
                        device.destroy_framebuffer(framebuffer_desc.framebuffer, None);
                    }
                }

                device.destroy_image_view(self.scene_image_view, None);
                device.destroy_image(self.scene_image, None);
                device.free_memory(self.scene_image_memory, None);
                device.destroy_buffer(self.scene_staging_buffer, None);
                device.free_memory(self.scene_staging_buffer_memory, None);
                device.destroy_sampler(self.scene_sampler, None);
                device.destroy_buffer(self.scene_vertex_buffer, None);
                device.free_memory(self.scene_vertex_buffer_memory, None);
                for eye in 0..2 {
                    device.destroy_buffer(self.scene_constant_buffer[eye], None);
                    device.free_memory(self.scene_constant_buffer_memory[eye], None);
                }

                device.destroy_buffer(self.companion_window_vertex_buffer, None);
                device.free_memory(self.companion_window_vertex_buffer_memory, None);
                device.destroy_buffer(self.companion_window_index_buffer, None);
                device.free_memory(self.companion_window_index_buffer_memory, None);

                device.destroy_buffer(self.controller_axes_vertex_buffer, None);
                device.free_memory(self.controller_axes_vertex_buffer_memory, None);

                device.destroy_pipeline_layout(self.pipeline_layout, None);
                device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

                for pso in 0..PSO_COUNT {
                    device.destroy_pipeline(self.pipelines[pso], None);
                }
                for shader in &self.shader_modules {
                    device.destroy_shader_module(*shader, None);
                }
                device.destroy_pipeline_cache(self.pipeline_cache, None);

                if let Some(destroy_debug_callback) = self.vk_destroy_debug_report_callback_ext {
                    destroy_debug_callback(
                        self.instance.as_ref().unwrap().handle(),
                        self.debug_report_callback,
                        ptr::null(),
                    );
                }

                for swapchain_index in 0..self.swapchain_framebuffers.len() {
                    device.destroy_framebuffer(self.swapchain_framebuffers[swapchain_index], None);
                    device.destroy_image_view(self.swapchain_image_views[swapchain_index], None);
                    device.destroy_semaphore(self.swapchain_semaphores[swapchain_index], None);
                }
                device.destroy_render_pass(self.swapchain_render_pass, None);

                self.swapchain_loader
                    .as_ref()
                    .unwrap()
                    .destroy_swapchain(self.swapchain, None);
                self.surface_loader
                    .as_ref()
                    .unwrap()
                    .destroy_surface(self.surface, None);

                self.device.as_ref().unwrap().destroy_device(None);
                self.instance.as_ref().unwrap().destroy_instance(None);

                // XXX: Window shutdown moved to window thread.
            }
        }
        return Ok(());
    }

    // XXX: OpenVR and Winit both want to control the main loop, so they've been split into two
    // threads.
    fn handle_vr_input(&mut self) -> bool {
        // XXX: We use openvr_sys here since the openvr crate appears to have uninitialized memory
        // bugs for these functions.
        let system_table: *const openvr_sys::VR_IVRSystem_FnTable =
            openvr_sys_load(openvr_sys::IVRSystem_Version).unwrap();
        let mut event = openvr_sys::VREvent_t {
            eventType: 0,
            trackedDeviceIndex: 0,
            eventAgeSeconds: 0.,
            data: openvr_sys::VREvent_Data_t {
                reserved: openvr_sys::VREvent_Reserved_t {
                    reserved0: 0,
                    reserved1: 0,
                    reserved2: 0,
                    reserved3: 0,
                    reserved4: 0,
                    reserved5: 0,
                },
            },
        };
        while unsafe {
            (*system_table)
                .PollNextEvent
                .map(|f| {
                    f(
                        &mut event as *mut openvr_sys::VREvent_t,
                        mem::size_of::<openvr_sys::VREvent_t>() as u32,
                    )
                })
                .unwrap_or(false)
        } {
            self.process_vr_event(event.into());
        }
        for device in 0..openvr::MAX_TRACKED_DEVICE_COUNT {
            let mut state = openvr::ControllerState {
                packet_num: 0,
                button_pressed: 0,
                button_touched: 0,
                axis: [openvr::ControllerAxis { x: 0., y: 0. }; 5],
            };
            unsafe {
                if (*system_table)
                    .GetControllerState
                    .map(|f| {
                        f(
                            device as u32,
                            &mut state as *mut openvr::ControllerState
                                as *mut openvr_sys::VRControllerState_t,
                            mem::size_of::<openvr::ControllerState>() as u32,
                        )
                    })
                    .unwrap_or(false)
                {
                    self.show_tracked_device[device] = state.button_pressed == 0;
                }
            }
        }
        return false;
    }

    // XXX: This function doesn't take self, since when it's called self is exclusively being used
    // by the vr thread
    fn handle_input(
        event: &winit::event::Event<()>,
        device: &ash::Device,
        rebuild_swapchain: &atomic::AtomicBool,
    ) -> bool {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    unsafe { device.device_wait_idle().unwrap() };
                    return true;
                }
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                            unsafe { device.device_wait_idle().unwrap() };
                            return true;
                        }
                        _ => {}
                    },
                },
                WindowEvent::Resized(_new_size) => {
                    // XXX: This code path doesn't exist in the original; it just crashes if the
                    // swapchain needs to be rebuilt.
                    rebuild_swapchain.store(true, atomic::Ordering::SeqCst);
                }
                _ => {}
            },
            Event::MainEventsCleared => {}
            Event::RedrawRequested(_window_id) => {}
            Event::LoopDestroyed => {
                unsafe { device.device_wait_idle().unwrap() };
            }
            _ => (),
        };
        return false;
    }

    // XXX: This function is very different from the original
    fn run_main_loop(&mut self) -> Result<(), Error> {
        let quit = Arc::new(AtomicBool::new(false));
        let rebuild_swapchain = Arc::new(AtomicBool::new(false));
        let quit_vr = Arc::clone(&quit);
        let rebuild_swapchain_vr = Arc::clone(&rebuild_swapchain);
        let device = self.device.as_ref().unwrap().clone();
        let mut event_loop = self
            .event_loop
            .take()
            .ok_or_else(|| err_msg("Could not get event loop"))?;
        let this = UnsafeSend::new(self as *mut MainApplication);
        // Ensure only a single &mut self exists at once.
        drop(self);
        let vr_thread = thread::spawn(move || {
            // This is safe because the only un-sync field (event_loop) is None.
            // The lifetime of `this` doesn't overlap with &mut self due to the drop above and join
            // below.
            let this: &mut MainApplication = unsafe { &mut *this.unwrap() };
            assert!(this.event_loop.is_none());
            while !quit_vr.load(std::sync::atomic::Ordering::SeqCst) {
                if this.handle_vr_input() {
                    quit_vr.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                // XXX: The original assumes that the swapchain will never need to
                if rebuild_swapchain_vr.load(atomic::Ordering::SeqCst) {
                    rebuild_swapchain_vr.store(false, atomic::Ordering::SeqCst);
                    unsafe {
                        let device = this.device.as_ref().unwrap();
                        device.queue_wait_idle(this.queue).unwrap();
                        for swapchain_index in 0..this.swapchain_framebuffers.len() {
                            device.destroy_framebuffer(
                                this.swapchain_framebuffers[swapchain_index],
                                None,
                            );
                            device.destroy_image_view(
                                this.swapchain_image_views[swapchain_index],
                                None,
                            );
                            device.destroy_semaphore(
                                this.swapchain_semaphores[swapchain_index],
                                None,
                            );
                        }
                        this.swapchain_framebuffers.clear();
                        this.swapchain_image_views.clear();
                        this.swapchain_semaphores.clear();
                        device.destroy_render_pass(this.swapchain_render_pass, None);
                        this.swapchain_loader
                            .as_ref()
                            .unwrap()
                            .destroy_swapchain(this.swapchain, None);
                        this.surface_loader
                            .as_ref()
                            .unwrap()
                            .destroy_surface(this.surface, None);
                    }
                    this.init_vulkan_swapchain().unwrap();
                }
                this.render_frame().unwrap();
            }
        });
        event_loop.run_return(move |event, _, ctrl_flow| {
            if MainApplication::handle_input(&event, &device, rebuild_swapchain.as_ref()) {
                quit.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            if quit.load(std::sync::atomic::Ordering::SeqCst) {
                *ctrl_flow = ControlFlow::Exit;
            }
        });
        vr_thread.join().expect("VR thread crashed");
        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Processes a single VR event
    //-----------------------------------------------------------------------------
    fn process_vr_event(&mut self, event: openvr::system::event::EventInfo) {
        match &event.event {
            openvr::system::event::Event::TrackedDeviceActivated => {
                self.setup_render_model_for_tracked_device(event.tracked_device_index)
                    .unwrap();
                dprintln!(
                    "Device {} attached. Setting up render model.",
                    event.tracked_device_index
                );
            }
            openvr::system::event::Event::TrackedDeviceDeactivated => {
                dprintln!("Device {} detached.", event.tracked_device_index);
            }
            openvr::system::event::Event::TrackedDeviceUpdated => {
                dprintln!("Device {} updated.", event.tracked_device_index);
            }
            _ => {}
        }
    }

    fn render_frame(&mut self) -> Result<(), Error> {
        if self.hmd.is_some() {
            self.current_command_buffer = self.get_command_buffer()?;

            // Start the command buffer
            {
                let device = self.device.as_ref().unwrap();

                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                unsafe {
                    device.begin_command_buffer(
                        self.current_command_buffer.command_buffer.clone(),
                        &command_buffer_begin_info,
                    )
                }?;
            }

            self.update_controller_axes()?;
            self.render_stereo_targets()?;
            self.render_companion_window()?;

            {
                let device = self.device.as_ref().unwrap();
                // End the command buffer
                unsafe {
                    device.end_command_buffer(self.current_command_buffer.command_buffer.clone())
                }?;
                // Submit the command buffer
                let command_buffers = [self.current_command_buffer.command_buffer.clone()];
                let wait_semaphores = [self.swapchain_semaphores[self.frame_index].clone()];
                let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let submit_info = [vk::SubmitInfo::builder()
                    .command_buffers(&command_buffers[..])
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_dst_stage_mask)
                    .build()];

                unsafe {
                    device.queue_submit(self.queue, &submit_info, self.current_command_buffer.fence)
                }?;
                unsafe { device.queue_wait_idle(self.queue) }?;
                // Add the command buffer back for later recycling
                self.command_buffers
                    .push_front(self.current_command_buffer.clone());
                self.current_command_buffer.command_buffer = vk::CommandBuffer::null();
                self.current_command_buffer.fence = vk::Fence::null();
            }

            // Submit to SteamVR
            let bounds = openvr::compositor::texture::Bounds {
                min: (0., 0.),
                max: (1., 1.),
            };

            let color_space = openvr::compositor::texture::ColorSpace::Auto;
            let mut vulkan_data = openvr::compositor::texture::vulkan::Texture {
                image: self.left_eye_desc.image.as_raw(),
                device: self.device.as_ref().unwrap().handle().as_raw() as *mut openvr::VkDevice_T,
                physical_device: self.physical_device.as_raw() as *mut openvr::VkPhysicalDevice_T,
                instance: self.instance.as_ref().unwrap().handle().as_raw()
                    as *mut openvr::VkInstance_T,
                queue: self.queue.as_raw() as *mut openvr::VkQueue_T,
                queue_family_index: self.queue_family_index,
                width: self.render_width,
                height: self.render_height,
                format: vk::Format::R8G8B8A8_SRGB.as_raw() as u32,
                sample_count: self.msaa_sample_count,
            };
            let left_eye_texture = openvr::compositor::texture::Texture {
                handle: openvr::compositor::texture::Handle::Vulkan(vulkan_data.clone()),
                color_space,
            };
            unsafe {
                if let Err(err) = self.compositor.as_ref().unwrap().submit(
                    openvr::Eye::Left,
                    &left_eye_texture,
                    Some(&bounds),
                    None,
                ) {
                    dprintln!("Error submitting eye: {:?}", err);
                }
            }
            vulkan_data.image = self.right_eye_desc.image.as_raw();
            let right_eye_texture = openvr::compositor::texture::Texture {
                handle: openvr::compositor::texture::Handle::Vulkan(vulkan_data.clone()),
                color_space,
            };
            unsafe {
                if let Err(err) = self.compositor.as_ref().unwrap().submit(
                    openvr::Eye::Right,
                    &right_eye_texture,
                    Some(&bounds),
                    None,
                ) {
                    dprintln!("Error submitting eye: {:?}", err);
                }
            }
        } else {
            dprintln!("no hmd");
        }

        let swapchains = [self.swapchain.clone()];
        let current_swapchain_image = [self.current_swapchain_image];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .image_indices(&current_swapchain_image);
        unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .queue_present(self.queue, &present_info)
        }?;

        // Spew out the controller and pose count whenever they change.
        if self.tracked_controller_count != self.tracked_controller_count_last
            || self.valid_pose_count != self.valid_pose_count_last
        {
            self.valid_pose_count_last = self.valid_pose_count;
            self.tracked_controller_count_last = self.tracked_controller_count;
            dprintln!(
                "PoseCount:{}({}) Controllers:{}",
                self.valid_pose_count,
                self.pose_classes,
                self.tracked_controller_count
            );
        }

        self.update_hmd_matrix_pose()?;

        self.frame_index = (self.frame_index + 1) % self.swapchain_images.len();

        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Creates all the shaders used by HelloVR Vulkan
    //-----------------------------------------------------------------------------
    fn create_all_shaders(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        // XXX: Some path logic was moved to acquire_file()
        let shader_names = ["scene", "axes", "rendermodel", "companion"];

        let stage_names = ["vs", "ps"];

        let mut shader_path = PathBuf::from("shaders");

        // Load the SPIR-V into shader modules
        for (shader_index, shader_name) in shader_names.iter().enumerate() {
            for (stage_index, stage_name) in stage_names.iter().enumerate() {
                shader_path.push(format!("{}_{}.spv", shader_name, stage_name));

                let mut f = acquire_file(&shader_path)?;
                shader_path.pop();

                let size = f.metadata().unwrap().len() as usize;
                assert!(size % mem::size_of::<u32>() == 0);
                let mut buffer = vec![0; size / mem::size_of::<u32>()];
                f.read_u32_into::<byteorder::NativeEndian>(&mut buffer)?;
                let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&buffer);
                self.shader_modules[shader_index * 2 + stage_index] =
                    unsafe { device.create_shader_module(&shader_module_create_info, None) }?;
            }
        }

        // Create a descriptor set layout/pipeline layout compatible with all of our shaders.  See bin/shaders/build_vulkan_shaders.bat for
        // how the HLSL is compiled with glslangValidator and binding numbers are generated

        let layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLER,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: ptr::null(),
            },
        ];
        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);
        self.descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        }?;

        let layouts = [self.descriptor_set_layout.clone()];

        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
        self.pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }?;

        // Create pipeline cache
        self.pipeline_cache =
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder(), None) }?;

        // Renderpass for each PSO that is compatible with what it will render to
        let render_passes = [
            self.left_eye_desc.render_pass,
            self.left_eye_desc.render_pass,
            self.left_eye_desc.render_pass,
            self.swapchain_render_pass,
        ];

        let strides = [
            mem::size_of::<VertexDataScene>(),               // PSO_SCENE
            mem::size_of::<f32>() * 6,                       // PSO_AXES
            mem::size_of::<openvr::render_models::Vertex>(), // PSO_RENDERMODEL
            mem::size_of::<VertexDataWindow>(),              // PSO_COMPANION
        ];

        let attribute_descriptions = [
            // PSO_SCENE
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: memoffset::offset_of!(VertexDataScene, tex_coord) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::UNDEFINED,
                offset: 0,
            },
            // PSO_AXES
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: mem::size_of::<f32>() as u32 * 3,
            },
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::UNDEFINED,
                offset: 0,
            },
            // PSO_RENDERMODEL
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: memoffset::offset_of!(openvr::render_models::Vertex, normal) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: memoffset::offset_of!(openvr::render_models::Vertex, texture_coord) as u32,
            },
            // PSO_COMPANION
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: mem::size_of::<f32>() as u32 * 2,
            },
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::UNDEFINED,
                offset: 0,
            },
        ];

        // Create the PSOs
        for pso_index in 0..PSO_COUNT {
            let binding_description = [vk::VertexInputBindingDescription {
                binding: 0,
                input_rate: vk::VertexInputRate::VERTEX,
                stride: strides[pso_index] as u32,
            }];

            let mut vertext_attribute_description_count = 0;
            for attr_index in 0..3 {
                if attribute_descriptions[pso_index * 3 + attr_index].format
                    != vk::Format::UNDEFINED
                {
                    vertext_attribute_description_count += 1;
                }
            }

            let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(
                    &attribute_descriptions
                        [3 * pso_index..3 * pso_index + vertext_attribute_description_count],
                )
                .vertex_binding_descriptions(&binding_description);

            let ds_state = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(pso_index != PSO_COMPANION)
                .depth_write_enable(pso_index != PSO_COMPANION)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .min_depth_bounds(0.0)
                .max_depth_bounds(0.0);

            let cb_attachment_state = [vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::from_raw(0xf))
                .build()];
            let cb_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(&cb_attachment_state);

            let rs_state = vk::PipelineRasterizationStateCreateInfo::builder()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);

            let ia_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(if pso_index == PSO_AXES {
                    vk::PrimitiveTopology::LINE_LIST
                } else {
                    vk::PrimitiveTopology::TRIANGLE_LIST
                })
                .primitive_restart_enable(false);

            let sample_mask = [0xFFFFFFFF];
            let ms_state = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(if pso_index == PSO_COMPANION {
                    vk::SampleCountFlags::TYPE_1
                } else {
                    vk::SampleCountFlags::from_raw(self.msaa_sample_count)
                })
                .min_sample_shading(0.0)
                .sample_mask(&sample_mask);

            let vp_state = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(self.shader_modules[pso_index * 2])
                    .name(cstr!("VSMain"))
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(self.shader_modules[pso_index * 2 + 1])
                    .name(cstr!("PSMain"))
                    .build(),
            ];

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

            let dynamic_state_create_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

            let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
                .layout(self.pipeline_layout)
                .vertex_input_state(&vertex_input_create_info)
                .input_assembly_state(&ia_state)
                .rasterization_state(&rs_state)
                .viewport_state(&vp_state)
                .multisample_state(&ms_state)
                .depth_stencil_state(&ds_state)
                .color_blend_state(&cb_state)
                .stages(&shader_stages)
                .render_pass(render_passes[pso_index])
                .dynamic_state(&dynamic_state_create_info)
                .build()];

            let pipelines = unsafe {
                device.create_graphics_pipelines(
                    self.pipeline_cache.clone(),
                    &pipeline_create_info,
                    None,
                )
            }
            .map_err(|e| e.1)?;
            assert!(pipelines.len() == 1);

            self.pipelines[pso_index] = pipelines[0];
        }

        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Creates all the descriptor sets
    //-----------------------------------------------------------------------------
    fn create_all_descriptor_sets(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                descriptor_count: NUM_DESCRIPTOR_SETS as u32,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
            },
            vk::DescriptorPoolSize {
                descriptor_count: NUM_DESCRIPTOR_SETS as u32,
                ty: vk::DescriptorType::SAMPLED_IMAGE,
            },
            vk::DescriptorPoolSize {
                descriptor_count: NUM_DESCRIPTOR_SETS as u32,
                ty: vk::DescriptorType::SAMPLER,
            },
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .max_sets(NUM_DESCRIPTOR_SETS as u32)
            .pool_sizes(&pool_sizes);

        self.descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }?;

        for descriptor_set in 0..NUM_DESCRIPTOR_SETS {
            let layouts = [self.descriptor_set_layout.clone()];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.descriptor_pool.clone())
                .set_layouts(&layouts);
            self.descriptor_sets[descriptor_set] =
                unsafe { device.allocate_descriptor_sets(&alloc_info) }.map(|v| {
                    assert!(v.len() == 1);
                    v[0]
                })?;
        }

        // Scene descriptor sets
        for eye_index in 0..2 {
            let buffer_info = [vk::DescriptorBufferInfo::builder()
                .buffer(self.scene_constant_buffer[eye_index].clone())
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()];
            let image_info = [vk::DescriptorImageInfo::builder()
                .image_view(self.scene_image_view.clone())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()];
            let sampler_info = [vk::DescriptorImageInfo::builder()
                .sampler(self.scene_sampler.clone())
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[DESCRIPTOR_SET_LEFT_EYE_SCENE + eye_index])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[DESCRIPTOR_SET_LEFT_EYE_SCENE + eye_index])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[DESCRIPTOR_SET_LEFT_EYE_SCENE + eye_index])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_info)
                    .build(),
            ];
            unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };
        }

        // Companion window descriptor sets
        {
            let mut image_info = [vk::DescriptorImageInfo::builder()
                .image_view(self.left_eye_desc.image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()];
            let mut write_descriptor_sets = [vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[DESCRIPTOR_SET_COMPANION_LEFT_TEXTURE])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_info)
                .build()];

            unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };
            image_info[0].image_view = self.right_eye_desc.image_view;
            write_descriptor_sets[0].dst_set =
                self.descriptor_sets[DESCRIPTOR_SET_COMPANION_RIGHT_TEXTURE];
            unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };
        }
        Ok(())
    }

    fn setup_texture_maps(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();

        // XXX: Some path logic was moved to acquire_file()
        let mut f = acquire_file(&Path::new("cube_texture.png"))?;

        let size = f.metadata().unwrap().len() as usize;
        let mut f_contents = Vec::with_capacity(size);
        f.read_to_end(&mut f_contents)?;
        assert!(f_contents.len() == size);

        let image = ImageReader::new(Cursor::new(&f_contents))
            .with_guessed_format()?
            .decode()?;

        let channels: u32 = 4;
        let image_rgba = image.to_rgba8();

        let (image_width, image_height) = image_rgba.dimensions();

        // Copy the base level to a buffer, reserve space for mips (overreserve by a bit to avoid having to calc mipchain size ahead of time)
        let mut buffer = vec![0; (image_width * image_height * channels * 2) as usize];
        buffer[..(image_width * image_height * channels) as usize]
            .copy_from_slice(image_rgba.as_flat_samples().as_slice());

        // XXX: pointers replaced with indices so we can use split_at_mut
        let mut prev_start = 0;
        let mut cur_start = (image_width * image_height * channels) as usize;

        let mut buffer_image_copies = Vec::new();

        let mut buffer_image_copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                base_array_layer: 0,
                layer_count: 1,
                mip_level: 0,
                aspect_mask: vk::ImageAspectFlags::COLOR,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: image_width,
                height: image_height,
                depth: 1,
            },
        };

        buffer_image_copies.push(buffer_image_copy.clone());

        let mut mip_width = image_width;
        let mut mip_height = image_height;

        while mip_width > 1 && mip_height > 1 {
            let (src, dst) = buffer.split_at_mut(cur_start as usize);
            let src = &mut src[prev_start..];

            let (new_mip_width, new_mip_height) =
                Self::gen_mip_map_rgba(src, dst, mip_width, mip_height);
            mip_width = new_mip_width;
            mip_height = new_mip_height;
            buffer_image_copy.buffer_offset = cur_start as u64;
            buffer_image_copy.image_subresource.mip_level += 1;
            buffer_image_copy.image_extent.width = mip_width;
            buffer_image_copy.image_extent.height = mip_height;
            buffer_image_copies.push(buffer_image_copy.clone());
            prev_start = cur_start;
            cur_start += (mip_width * mip_height * channels) as usize;
        }

        let buffer_size: vk::DeviceSize = cur_start as u64;

        // Create the image
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: image_width,
                height: image_height,
                depth: 1,
            })
            .mip_levels(buffer_image_copies.len() as u32)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .flags(vk::ImageCreateFlags::empty())
            .build();

        self.scene_image = unsafe { device.create_image(&image_create_info, None) }?;

        let memory_requirements = unsafe { device.get_image_memory_requirements(self.scene_image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                memory_type_from_properties(
                    &self.physical_device_memory_properties,
                    memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or(err_msg("Could not get memory type index"))?,
            );

        self.scene_image_memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }?;
        unsafe { device.bind_image_memory(self.scene_image, self.scene_image_memory, 0) }?;

        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(self.scene_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image_create_info.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: image_create_info.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });
        self.scene_image_view = unsafe { device.create_image_view(&image_view_create_info, None) }?;

        // Create a staging buffer
        let vulkan_buffer = create_vulkan_buffer(
            device,
            &self.physical_device_memory_properties,
            Some(&buffer),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        self.scene_staging_buffer = vulkan_buffer.0;
        self.scene_staging_buffer_memory = vulkan_buffer.1;

        // Transition the image to TRANSFER_DST to receive image
        let mut image_memory_barrier = [vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(self.scene_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: image_create_info.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .build()];

        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }

        // Issue the copy to fill the image data
        unsafe {
            device.cmd_copy_buffer_to_image(
                self.current_command_buffer.command_buffer,
                self.scene_staging_buffer,
                self.scene_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_image_copies,
            );
        }

        // Transition the image to SHADER_READ_OPTIMAL for reading
        image_memory_barrier[0].src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        image_memory_barrier[0].dst_access_mask = vk::AccessFlags::SHADER_READ;
        image_memory_barrier[0].old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        image_memory_barrier[0].new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }

        // Create the sampler
        let sampler_create_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .min_lod(0.0)
            .max_lod(image_create_info.mip_levels as f32);

        self.scene_sampler = unsafe { device.create_sampler(&sampler_create_info, None) }?;

        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: generate next level mipmap for an RGBA image
    //-----------------------------------------------------------------------------
    fn gen_mip_map_rgba(src: &[u8], dst: &mut [u8], src_width: u32, src_height: u32) -> (u32, u32) {
        let dst_width = u32::max(1, src_width / 2);
        let dst_height = u32::max(1, src_height / 2);

        for y in 0..dst_height {
            for x in 0..dst_width {
                let mut r = 0.0;
                let mut g = 0.0;
                let mut b = 0.0;
                let mut a = 0.0;

                let src_index: [usize; 4] = [
                    (((y * 2) * src_width) + (x * 2)) as usize * 4,
                    (((y * 2) * src_width) + (x * 2 + 1)) as usize * 4,
                    ((((y * 2) + 1) * src_width) + (x * 2)) as usize * 4,
                    ((((y * 2) + 1) * src_width) + (x * 2 + 1)) as usize * 4,
                ];

                // Sum all pixels
                for sample in 0..4 {
                    r += src[src_index[sample]] as f32;
                    g += src[src_index[sample] + 1] as f32;
                    b += src[src_index[sample] + 2] as f32;
                    a += src[src_index[sample] + 3] as f32;
                }

                // Average results
                r /= 4.0;
                g /= 4.0;
                b /= 4.0;
                a /= 4.0;

                // Store resulting pixels
                dst[(y * dst_width + x) as usize * 4] = r as u8;
                dst[(y * dst_width + x) as usize * 4 + 1] = g as u8;
                dst[(y * dst_width + x) as usize * 4 + 2] = b as u8;
                dst[(y * dst_width + x) as usize * 4 + 3] = a as u8;
            }
        }
        return (dst_width, dst_height);
    }

    //-----------------------------------------------------------------------------
    // Purpose: create a sea of cubes
    //-----------------------------------------------------------------------------
    fn setup_scene(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();

        if self.hmd.is_none() {
            return Err(err_msg("No hmd"));
        }

        let mut vert_data: Vec<f32> = Vec::new();

        let mut mat = Matrix4::identity()
            .append_scaling(self.scale)
            .append_translation(&Vector3::new(
                -self.scene_volume_width as f32 * self.scale_spacing / 2.,
                -self.scene_volume_height as f32 * self.scale_spacing / 2.,
                -self.scene_volume_depth as f32 * self.scale_spacing / 2.,
            ));

        for _z in 0..self.scene_volume_depth {
            for _y in 0..self.scene_volume_height {
                for _x in 0..self.scene_volume_width {
                    Self::add_cube_to_scene(mat, &mut vert_data);
                    mat = mat.append_translation(&Vector3::new(self.scale_spacing, 0., 0.));
                }
                mat = mat.append_translation(&Vector3::new(
                    -self.scene_volume_width as f32 * self.scale_spacing,
                    self.scale_spacing,
                    0.,
                ));
            }
            mat = mat.append_translation(&Vector3::new(
                0.,
                -self.scene_volume_height as f32 * self.scale_spacing,
                self.scale_spacing,
            ));
        }
        // XXX: This 5 is because each vert has x,y,z,u,v
        self.vert_count = vert_data.len() as u32 / 5;

        // Create the vertex buffer and fill with data
        let vulkan_buffer = create_vulkan_buffer(
            device,
            &self.physical_device_memory_properties,
            Some(&vert_data),
            (vert_data.len() * mem::size_of::<f32>()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        self.scene_vertex_buffer = vulkan_buffer.0;
        self.scene_vertex_buffer_memory = vulkan_buffer.1;

        // Create constant buffer to hold the per-eye CB data
        for eye_index in 0..2 {
            assert!(mem::size_of::<Matrix4<f32>>() == mem::size_of::<f32>() * 4 * 4);
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(mem::size_of::<Matrix4<f32>>() as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);
            self.scene_constant_buffer[eye_index] =
                unsafe { device.create_buffer(&buffer_create_info, None) }?;
            let memory_requirements = unsafe {
                device.get_buffer_memory_requirements(self.scene_constant_buffer[eye_index])
            };
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .memory_type_index(
                    memory_type_from_properties(
                        &self.physical_device_memory_properties,
                        memory_requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT
                            | vk::MemoryPropertyFlags::HOST_CACHED,
                    )
                    .ok_or(err_msg("Could not get memory type index"))?,
                )
                .allocation_size(memory_requirements.size);

            self.scene_constant_buffer_memory[eye_index] =
                unsafe { device.allocate_memory(&alloc_info, None) }?;
            unsafe {
                device.bind_buffer_memory(
                    self.scene_constant_buffer[eye_index],
                    self.scene_constant_buffer_memory[eye_index],
                    0,
                )
            }?;

            // Map and keep mapped persistently
            self.scene_constant_buffer_data[eye_index] = unsafe {
                device.map_memory(
                    self.scene_constant_buffer_memory[eye_index],
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            }? as *mut Matrix4<f32>;
        }
        return Ok(());
    }

    fn add_cube_vertex(fl0: f32, fl1: f32, fl2: f32, fl3: f32, fl4: f32, vert_data: &mut Vec<f32>) {
        vert_data.push(fl0);
        vert_data.push(fl1);
        vert_data.push(fl2);
        vert_data.push(fl3);
        vert_data.push(fl4);
    }

    fn add_cube_to_scene(mat: Matrix4<f32>, vert_data: &mut Vec<f32>) {
        let a = mat * Vector4::new(0., 0., 0., 1.);
        let b = mat * Vector4::new(1., 0., 0., 1.);
        let c = mat * Vector4::new(1., 1., 0., 1.);
        let d = mat * Vector4::new(0., 1., 0., 1.);
        let e = mat * Vector4::new(0., 0., 1., 1.);
        let f = mat * Vector4::new(1., 0., 1., 1.);
        let g = mat * Vector4::new(1., 1., 1., 1.);
        let h = mat * Vector4::new(0., 1., 1., 1.);

        // triangles instead of quads
        Self::add_cube_vertex(e[0], e[1], e[2], 0., 1., vert_data); //Front
        Self::add_cube_vertex(f[0], f[1], f[2], 1., 1., vert_data);
        Self::add_cube_vertex(g[0], g[1], g[2], 1., 0., vert_data);
        Self::add_cube_vertex(g[0], g[1], g[2], 1., 0., vert_data);
        Self::add_cube_vertex(h[0], h[1], h[2], 0., 0., vert_data);
        Self::add_cube_vertex(e[0], e[1], e[2], 0., 1., vert_data);

        Self::add_cube_vertex(b[0], b[1], b[2], 0., 1., vert_data); //Back
        Self::add_cube_vertex(a[0], a[1], a[2], 1., 1., vert_data);
        Self::add_cube_vertex(d[0], d[1], d[2], 1., 0., vert_data);
        Self::add_cube_vertex(d[0], d[1], d[2], 1., 0., vert_data);
        Self::add_cube_vertex(c[0], c[1], c[2], 0., 0., vert_data);
        Self::add_cube_vertex(b[0], b[1], b[2], 0., 1., vert_data);

        Self::add_cube_vertex(h[0], h[1], h[2], 0., 1., vert_data); //Top
        Self::add_cube_vertex(g[0], g[1], g[2], 1., 1., vert_data);
        Self::add_cube_vertex(c[0], c[1], c[2], 1., 0., vert_data);
        Self::add_cube_vertex(c[0], c[1], c[2], 1., 0., vert_data);
        Self::add_cube_vertex(d[0], d[1], d[2], 0., 0., vert_data);
        Self::add_cube_vertex(h[0], h[1], h[2], 0., 1., vert_data);

        Self::add_cube_vertex(a[0], a[1], a[2], 0., 1., vert_data); //Bottom
        Self::add_cube_vertex(b[0], b[1], b[2], 1., 1., vert_data);
        Self::add_cube_vertex(f[0], f[1], f[2], 1., 0., vert_data);
        Self::add_cube_vertex(f[0], f[1], f[2], 1., 0., vert_data);
        Self::add_cube_vertex(e[0], e[1], e[2], 0., 0., vert_data);
        Self::add_cube_vertex(a[0], a[1], a[2], 0., 1., vert_data);

        Self::add_cube_vertex(a[0], a[1], a[2], 0., 1., vert_data); //Left
        Self::add_cube_vertex(e[0], e[1], e[2], 1., 1., vert_data);
        Self::add_cube_vertex(h[0], h[1], h[2], 1., 0., vert_data);
        Self::add_cube_vertex(h[0], h[1], h[2], 1., 0., vert_data);
        Self::add_cube_vertex(d[0], d[1], d[2], 0., 0., vert_data);
        Self::add_cube_vertex(a[0], a[1], a[2], 0., 1., vert_data);

        Self::add_cube_vertex(f[0], f[1], f[2], 0., 1., vert_data); //Right
        Self::add_cube_vertex(b[0], b[1], b[2], 1., 1., vert_data);
        Self::add_cube_vertex(c[0], c[1], c[2], 1., 0., vert_data);
        Self::add_cube_vertex(c[0], c[1], c[2], 1., 0., vert_data);
        Self::add_cube_vertex(g[0], g[1], g[2], 0., 0., vert_data);
        Self::add_cube_vertex(f[0], f[1], f[2], 0., 1., vert_data);
    }

    //-----------------------------------------------------------------------------
    // Purpose: Update the vertex data for the controllers as X/Y/Z lines
    //-----------------------------------------------------------------------------
    fn update_controller_axes(&mut self) -> Result<(), Error> {
        // Don't attempt to update controllers if input is not available
        let system_table: *const openvr_sys::VR_IVRSystem_FnTable =
            openvr_sys_load(openvr_sys::IVRSystem_Version)?;
        unsafe {
            if !(*system_table)
                .IsInputAvailable
                .map(|f| f())
                .unwrap_or(false)
            {
                return Ok(());
            }
        }

        let mut vert_data = Vec::new();
        self.controller_vertcount = 0;
        self.tracked_controller_count = 0;

        let hmd = self.hmd.as_ref().unwrap();

        for tracked_device_index in 0..openvr::MAX_TRACKED_DEVICE_COUNT {
            if !hmd.is_tracked_device_connected(tracked_device_index as u32) {
                continue;
            }

            if hmd.tracked_device_class(tracked_device_index as u32)
                != openvr::TrackedDeviceClass::Controller
            {
                continue;
            }

            self.tracked_controller_count += 1;

            if !self.tracked_device_pose[tracked_device_index].pose_is_valid() {
                continue;
            }

            let mat = &self.device_pose[tracked_device_index];

            let center = mat * Vector4::new(0., 0., 0., 1.);

            for i in 0..3 {
                let mut color = Vector3::<f32>::zeros();
                let mut point = Vector4::new(0., 0., 0., 1.);
                point[i] += 0.05; // offset in X, Y, Z
                color[i] = 1.0; // R, G, B
                point = mat * point;
                vert_data.push(center[0]);
                vert_data.push(center[1]);
                vert_data.push(center[2]);
                vert_data.push(color[0]);
                vert_data.push(color[1]);
                vert_data.push(color[2]);
                vert_data.push(point[0]);
                vert_data.push(point[1]);
                vert_data.push(point[2]);
                vert_data.push(color[0]);
                vert_data.push(color[1]);
                vert_data.push(color[2]);
                self.controller_vertcount += 2;
            }
            let start = mat * Vector4::new(0., 0., -0.02, 1.);
            let end = mat * Vector4::new(0., 0., -39., 1.);
            let color = Vector3::new(0.92, 0.92, 0.71);

            vert_data.push(start[0]);
            vert_data.push(start[1]);
            vert_data.push(start[2]);

            vert_data.push(color[0]);
            vert_data.push(color[1]);
            vert_data.push(color[2]);

            vert_data.push(end[0]);
            vert_data.push(end[1]);
            vert_data.push(end[2]);

            vert_data.push(color[0]);
            vert_data.push(color[1]);
            vert_data.push(color[2]);
            self.controller_vertcount += 2;

            let device = self.device.as_ref().unwrap();

            // Setup the VB the first time through.
            if self.controller_axes_vertex_buffer == vk::Buffer::null() && vert_data.len() > 0 {
                // Make big enough to hold up to the max number
                let size =
                    (mem::size_of::<f32>() * vert_data.len()) * openvr::MAX_TRACKED_DEVICE_COUNT;
                let vulkan_buffer = create_vulkan_buffer::<()>(
                    self.device.as_ref().unwrap(),
                    &self.physical_device_memory_properties,
                    None,
                    size as u64,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                )?;

                self.controller_axes_vertex_buffer = vulkan_buffer.0;
                self.controller_axes_vertex_buffer_memory = vulkan_buffer.1;
            }

            // Update the VB data
            if self.controller_axes_vertex_buffer != vk::Buffer::null() && vert_data.len() > 0 {
                unsafe {
                    let data = device.map_memory(
                        self.controller_axes_vertex_buffer_memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )?;
                    data.copy_from_nonoverlapping(
                        vert_data.as_ptr() as *const c_void,
                        vert_data.len() * mem::size_of::<f32>(),
                    );
                    let memory_range = [vk::MappedMemoryRange::builder()
                        .memory(self.controller_axes_vertex_buffer_memory)
                        .size(vk::WHOLE_SIZE)
                        .build()];
                    device.flush_mapped_memory_ranges(&memory_range)?;
                    // XXX: Spec says we need to unmap *after* flushing.
                    device.unmap_memory(self.controller_axes_vertex_buffer_memory);
                }
            }
        }
        return Ok(());
    }

    fn setup_cameras(&mut self) {
        self.projection_left = self.get_hmd_matrix_projection_eye(openvr::Eye::Left);
        self.projection_right = self.get_hmd_matrix_projection_eye(openvr::Eye::Right);
        self.eye_pos_left = self.get_hmd_matrix_pose_eye(openvr::Eye::Left);
        self.eye_pos_right = self.get_hmd_matrix_pose_eye(openvr::Eye::Right);
    }

    //-----------------------------------------------------------------------------
    // Purpose: Creates a frame buffer. Returns true if the buffer was set up.
    //          Returns false if the setup failed.
    //-----------------------------------------------------------------------------
    fn create_frame_buffer(
        &mut self,
        width: u32,
        height: u32,
        framebuffer_desc: &mut FramebufferDesc,
    ) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();

        //---------------------------//
        //    Create color target    //
        //---------------------------//
        let mut image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .samples(vk::SampleCountFlags::from_raw(self.msaa_sample_count))
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .flags(vk::ImageCreateFlags::empty())
            .build();

        framebuffer_desc.image = unsafe { device.create_image(&image_create_info, None) }?;

        let memory_requirements =
            unsafe { device.get_image_memory_requirements(framebuffer_desc.image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                memory_type_from_properties(
                    &self.physical_device_memory_properties,
                    memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or(err_msg("Could not get memory type index"))?,
            );

        framebuffer_desc.device_memory =
            unsafe { device.allocate_memory(&memory_allocate_info, None) }?;

        unsafe {
            device.bind_image_memory(framebuffer_desc.image, framebuffer_desc.device_memory, 0)
        }?;

        let mut image_view_create_info = vk::ImageViewCreateInfo::builder()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(framebuffer_desc.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image_create_info.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        framebuffer_desc.image_view =
            unsafe { device.create_image_view(&image_view_create_info, None)? };

        //-----------------------------------//
        //    Create depth/stencil target    //
        //-----------------------------------//
        image_create_info.image_type = vk::ImageType::TYPE_2D;
        image_create_info.format = vk::Format::D32_SFLOAT;
        image_create_info.usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;

        framebuffer_desc.depth_stencil_image =
            unsafe { device.create_image(&image_create_info, None)? };

        let memory_requirements =
            unsafe { device.get_image_memory_requirements(framebuffer_desc.depth_stencil_image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                memory_type_from_properties(
                    &self.physical_device_memory_properties,
                    memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or(err_msg("Could not get memory type index"))?,
            );

        framebuffer_desc.depth_stencil_device_memory =
            unsafe { device.allocate_memory(&memory_allocate_info, None) }?;

        unsafe {
            device.bind_image_memory(
                framebuffer_desc.depth_stencil_image,
                framebuffer_desc.depth_stencil_device_memory,
                0,
            )
        }?;

        image_view_create_info.image = framebuffer_desc.depth_stencil_image;
        image_view_create_info.format = image_create_info.format;
        image_view_create_info.subresource_range.aspect_mask = vk::ImageAspectFlags::DEPTH;

        framebuffer_desc.depth_stencil_image_view =
            unsafe { device.create_image_view(&image_view_create_info, None) }?;

        // Create a renderpass
        let attachment_references = [
            vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
        ];
        let attachment_descs = [
            vk::AttachmentDescription {
                format: vk::Format::R8G8B8A8_SRGB,
                samples: image_create_info.samples,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                flags: vk::AttachmentDescriptionFlags::empty(),
            },
            vk::AttachmentDescription {
                format: vk::Format::D32_SFLOAT,
                samples: image_create_info.samples,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                flags: vk::AttachmentDescriptionFlags::empty(),
            },
        ];

        let sub_pass_create_info = [vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            flags: vk::SubpassDescriptionFlags::empty(),
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &attachment_references[0],
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: &attachment_references[1],
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        }];

        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .flags(vk::RenderPassCreateFlags::empty())
            .attachments(&attachment_descs)
            .subpasses(&sub_pass_create_info)
            .dependencies(&[]);

        framebuffer_desc.render_pass =
            unsafe { device.create_render_pass(&render_pass_create_info, None) }?;

        // Create the framebuffer
        let attachments = [
            framebuffer_desc.image_view,
            framebuffer_desc.depth_stencil_image_view,
        ];
        let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(framebuffer_desc.render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        framebuffer_desc.framebuffer =
            unsafe { device.create_framebuffer(&framebuffer_create_info, None) }?;

        framebuffer_desc.image_layout = vk::ImageLayout::UNDEFINED;
        framebuffer_desc.depth_stencil_image_layout = vk::ImageLayout::UNDEFINED;

        return Ok(());
    }

    fn setup_stereo_render_targets(&mut self) -> Result<(), Error> {
        if let Some(hmd) = self.hmd.as_ref() {
            let (width, height) = hmd.recommended_render_target_size();
            self.render_width = (self.super_sample_scale * width as f32) as u32;
            self.render_height = (self.super_sample_scale * height as f32) as u32;

            // XXX: Because the original uses multiple mutating pointers into self, we need to
            // workaround that here.
            let mut framebuffer_desc = FramebufferDesc::default();
            self.create_frame_buffer(self.render_width, self.render_height, &mut framebuffer_desc)?;
            self.left_eye_desc = framebuffer_desc;
            let mut framebuffer_desc = FramebufferDesc::default();
            self.create_frame_buffer(self.render_width, self.render_height, &mut framebuffer_desc)?;
            self.right_eye_desc = framebuffer_desc;
            return Ok(());
        } else {
            return Err(err_msg("No HMD"));
        }
    }

    fn setup_companion_window(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        if self.hmd.is_none() {
            return Ok(());
        }

        let mut verts = Vec::new();

        // left eye verts
        verts.push(VertexDataWindow::new(
            Vector2::new(-1., -1.),
            Vector2::new(0., 1.),
        ));
        verts.push(VertexDataWindow::new(
            Vector2::new(0., -1.),
            Vector2::new(1., 1.),
        ));
        verts.push(VertexDataWindow::new(
            Vector2::new(-1., 1.),
            Vector2::new(0., 0.),
        ));
        verts.push(VertexDataWindow::new(
            Vector2::new(0., 1.),
            Vector2::new(1., 0.),
        ));

        // right eye verts
        verts.push(VertexDataWindow::new(
            Vector2::new(0., -1.),
            Vector2::new(0., 1.),
        ));
        verts.push(VertexDataWindow::new(
            Vector2::new(1., -1.),
            Vector2::new(1., 1.),
        ));
        verts.push(VertexDataWindow::new(
            Vector2::new(0., 1.),
            Vector2::new(0., 0.),
        ));
        verts.push(VertexDataWindow::new(
            Vector2::new(1., 1.),
            Vector2::new(1., 0.),
        ));

        let vulkan_buffer = create_vulkan_buffer(
            device,
            &self.physical_device_memory_properties,
            Some(&verts),
            (mem::size_of::<VertexDataWindow>() * verts.len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        self.companion_window_vertex_buffer = vulkan_buffer.0;
        self.companion_window_vertex_buffer_memory = vulkan_buffer.1;

        // Create index buffer
        let indices = [0u16, 1, 3, 0, 3, 2, 4, 5, 7, 4, 7, 6];
        self.companion_window_index_size = indices.len() as u32;

        let vulkan_buffer = create_vulkan_buffer(
            device,
            &self.physical_device_memory_properties,
            Some(&indices),
            (mem::size_of::<u16>() * indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        self.companion_window_index_buffer = vulkan_buffer.0;
        self.companion_window_index_buffer_memory = vulkan_buffer.1;

        // Transition all of the swapchain images to PRESENT_SRC so they are ready for presentation
        for swapchain_image in &self.swapchain_images {
            let image_memory_barrier = [vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(swapchain_image.clone())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(self.queue_family_index)
                .build()];

            unsafe {
                device.cmd_pipeline_barrier(
                    self.current_command_buffer.command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_memory_barrier,
                );
            }
        }
        return Ok(());
    }

    fn render_stereo_targets(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        // Set viewport and scissor
        let viewport = [vk::Viewport {
            x: 0.,
            y: 0.,
            width: self.render_width as f32,
            height: self.render_height as f32,
            min_depth: 0.,
            max_depth: 1.,
        }];
        unsafe {
            device.cmd_set_viewport(self.current_command_buffer.command_buffer, 0, &viewport);
        }
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.render_width,
                height: self.render_height,
            },
        }];
        unsafe {
            device.cmd_set_scissor(self.current_command_buffer.command_buffer, 0, &scissor);
        }

        //----------//
        // Left Eye //
        //----------//
        let mut image_memory_barrier = [vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(self.left_eye_desc.image_layout)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.left_eye_desc.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .build()];

        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }

        self.left_eye_desc.image_layout = image_memory_barrier[0].new_layout;
        // Transition the depth buffer to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL on first use
        if self.left_eye_desc.depth_stencil_image_layout == vk::ImageLayout::UNDEFINED {
            image_memory_barrier[0].image = self.left_eye_desc.depth_stencil_image;
            image_memory_barrier[0].subresource_range.aspect_mask = vk::ImageAspectFlags::DEPTH;
            image_memory_barrier[0].src_access_mask = vk::AccessFlags::empty();
            image_memory_barrier[0].dst_access_mask =
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            image_memory_barrier[0].old_layout = self.left_eye_desc.depth_stencil_image_layout;
            image_memory_barrier[0].new_layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            unsafe {
                device.cmd_pipeline_barrier(
                    self.current_command_buffer.command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_memory_barrier,
                );
            }
            self.left_eye_desc.depth_stencil_image_layout = image_memory_barrier[0].new_layout;
        }

        // Start the renderpass
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.,
                    stencil: 0,
                },
            },
        ];
        let mut render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.left_eye_desc.render_pass)
            .framebuffer(self.left_eye_desc.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.render_width,
                    height: self.render_height,
                },
            })
            .clear_values(&clear_values)
            .build();

        unsafe {
            device.cmd_begin_render_pass(
                self.current_command_buffer.command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }

        // XXX: Can't keep a reference to device across method call
        drop(device);
        self.render_scene(openvr::Eye::Left)?;
        let device = self.device.as_ref().unwrap();

        unsafe {
            device.cmd_end_render_pass(self.current_command_buffer.command_buffer);
        }

        // Transition eye image to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for display on the companion window
        image_memory_barrier[0].image = self.left_eye_desc.image;
        image_memory_barrier[0].subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
        image_memory_barrier[0].src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        image_memory_barrier[0].dst_access_mask = vk::AccessFlags::SHADER_READ;
        image_memory_barrier[0].old_layout = self.left_eye_desc.image_layout;
        image_memory_barrier[0].new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }
        self.left_eye_desc.image_layout = image_memory_barrier[0].new_layout;

        //-----------//
        // Right Eye //
        //-----------//
        // Transition to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        image_memory_barrier[0].image = self.right_eye_desc.image;
        image_memory_barrier[0].subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
        image_memory_barrier[0].src_access_mask =
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ;
        image_memory_barrier[0].dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        image_memory_barrier[0].old_layout = self.right_eye_desc.image_layout;
        image_memory_barrier[0].new_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }
        self.right_eye_desc.image_layout = image_memory_barrier[0].new_layout;

        // Transition the depth buffer to VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL on first use
        if self.right_eye_desc.depth_stencil_image_layout == vk::ImageLayout::UNDEFINED {
            image_memory_barrier[0].image = self.right_eye_desc.depth_stencil_image;
            image_memory_barrier[0].subresource_range.aspect_mask = vk::ImageAspectFlags::DEPTH;
            image_memory_barrier[0].src_access_mask = vk::AccessFlags::empty();
            image_memory_barrier[0].dst_access_mask =
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            image_memory_barrier[0].old_layout = self.right_eye_desc.depth_stencil_image_layout;
            image_memory_barrier[0].new_layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            unsafe {
                device.cmd_pipeline_barrier(
                    self.current_command_buffer.command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_memory_barrier,
                );
            }
            self.right_eye_desc.depth_stencil_image_layout = image_memory_barrier[0].new_layout;
        }

        // Start the renderpass
        render_pass_begin_info.render_pass = self.right_eye_desc.render_pass;
        render_pass_begin_info.framebuffer = self.right_eye_desc.framebuffer;
        // XXX: This should do nothing?
        //assert!(render_pass_begin_info.p_clear_values == &clear_values[0]);
        render_pass_begin_info.p_clear_values = &clear_values[0];

        unsafe {
            device.cmd_begin_render_pass(
                self.current_command_buffer.command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }

        drop(device);
        self.render_scene(openvr::Eye::Right)?;
        let device = self.device.as_ref().unwrap();

        unsafe {
            device.cmd_end_render_pass(self.current_command_buffer.command_buffer);
        }

        // Transition eye image to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for display on the companion window
        image_memory_barrier[0].image = self.right_eye_desc.image;
        image_memory_barrier[0].subresource_range.aspect_mask = vk::ImageAspectFlags::COLOR;
        image_memory_barrier[0].src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        image_memory_barrier[0].dst_access_mask = vk::AccessFlags::SHADER_READ;
        image_memory_barrier[0].old_layout = self.right_eye_desc.image_layout;
        image_memory_barrier[0].new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }
        self.right_eye_desc.image_layout = image_memory_barrier[0].new_layout;

        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Renders a scene with respect to nEye.
    //-----------------------------------------------------------------------------
    fn render_scene(&mut self, eye: openvr::Eye) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        if self.show_cubes {
            unsafe {
                device.cmd_bind_pipeline(
                    self.current_command_buffer.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines[PSO_SCENE],
                );

                // Update the persistently mapped pointer to the CB data with the latest matrix
                // XXX: Replaced memcpy with typed dereference
                *self.scene_constant_buffer_data[eye as usize] =
                    self.get_current_view_projection_matrix(eye);

                device.cmd_bind_descriptor_sets(
                    self.current_command_buffer.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_sets[DESCRIPTOR_SET_LEFT_EYE_SCENE + eye as usize]],
                    &[],
                );
            }

            // Draw
            let offsets = [0];
            unsafe {
                device.cmd_bind_vertex_buffers(
                    self.current_command_buffer.command_buffer,
                    0,
                    &[self.scene_vertex_buffer],
                    &offsets,
                );
                device.cmd_draw(
                    self.current_command_buffer.command_buffer,
                    self.vert_count,
                    1,
                    0,
                    0,
                );
            }
        }

        // Don't attempt to update controllers if input is not available
        // XXX: Uses openvr_sys since the openvr crate doesn't expose this function.
        let system_table: *const openvr_sys::VR_IVRSystem_FnTable =
            openvr_sys_load(openvr_sys::IVRSystem_Version)?;
        let is_input_available = unsafe {
            (*system_table)
                .IsInputAvailable
                .map(|f| f())
                .unwrap_or(false)
        };
        if is_input_available && self.controller_axes_vertex_buffer != vk::Buffer::null() {
            // draw the controller axis lines
            unsafe {
                device.cmd_bind_pipeline(
                    self.current_command_buffer.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipelines[PSO_AXES],
                );

                let offsets = [0];
                device.cmd_bind_vertex_buffers(
                    self.current_command_buffer.command_buffer,
                    0,
                    &[self.controller_axes_vertex_buffer],
                    &offsets,
                );
                device.cmd_draw(
                    self.current_command_buffer.command_buffer,
                    self.controller_vertcount,
                    1,
                    0,
                    0,
                );
            }
        }

        // ----- Render Model rendering -----
        unsafe {
            device.cmd_bind_pipeline(
                self.current_command_buffer.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[PSO_RENDERMODEL],
            );
        }
        for tracked_device in 0..openvr::MAX_TRACKED_DEVICE_COUNT {
            if let Some(render_model_index) = self.tracked_device_to_render_model[tracked_device] {
                if !self.show_tracked_device[tracked_device] {
                    continue;
                }

                let pose = &self.tracked_device_pose[tracked_device];
                if !pose.pose_is_valid() {
                    continue;
                }

                if !is_input_available
                    && self
                        .hmd
                        .as_ref()
                        .unwrap()
                        .tracked_device_class(tracked_device as u32)
                        == openvr::TrackedDeviceClass::Controller
                {
                    continue;
                }

                let device_to_tracking = self.device_pose[tracked_device];
                let mvp = self.get_current_view_projection_matrix(eye) * device_to_tracking;

                self.render_models_vec[render_model_index].draw(
                    eye,
                    self.current_command_buffer.command_buffer,
                    self.pipeline_layout,
                    &mvp,
                )?;
            }
        }
        return Ok(());
    }

    fn render_companion_window(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        // Get the next swapchain image
        self.current_swapchain_image = unsafe {
            self.swapchain_loader.as_ref().unwrap().acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.swapchain_semaphores[self.frame_index],
                vk::Fence::null(),
            )
        }?
        .0;

        // Transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL for rendering
        let mut image_memory_barrier = [vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            // XXX: The swapchain might have been rebuilt, so don't specify the old layout
            //.old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.swapchain_images[self.current_swapchain_image as usize])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_queue_family_index(self.queue_family_index)
            .dst_queue_family_index(self.queue_family_index)
            .build()];

        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }

        // Start the renderpass
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.],
            },
        }];
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.swapchain_render_pass)
            .framebuffer(self.swapchain_framebuffers[self.current_swapchain_image as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.companion_window_width,
                    height: self.companion_window_height,
                },
            })
            .clear_values(&clear_values)
            .build();

        unsafe {
            device.cmd_begin_render_pass(
                self.current_command_buffer.command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }

        // Set viewport/scissor
        let viewport = [vk::Viewport {
            x: 0.,
            y: 0.,
            width: self.companion_window_width as f32,
            height: self.companion_window_height as f32,
            min_depth: 0.,
            max_depth: 1.,
        }];
        unsafe {
            device.cmd_set_viewport(self.current_command_buffer.command_buffer, 0, &viewport);
        }
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.companion_window_width,
                height: self.companion_window_height,
            },
        }];
        unsafe {
            device.cmd_set_scissor(self.current_command_buffer.command_buffer, 0, &scissor);
        }

        // Bind the pipeline and descriptor set
        unsafe {
            device.cmd_bind_pipeline(
                self.current_command_buffer.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[PSO_COMPANION],
            );

            device.cmd_bind_descriptor_sets(
                self.current_command_buffer.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[DESCRIPTOR_SET_COMPANION_LEFT_TEXTURE]],
                &[],
            );
        }

        // Draw left eye texture to companion window
        let offsets = [0];
        unsafe {
            device.cmd_bind_vertex_buffers(
                self.current_command_buffer.command_buffer,
                0,
                &[self.companion_window_vertex_buffer],
                &offsets,
            );
            device.cmd_bind_index_buffer(
                self.current_command_buffer.command_buffer,
                self.companion_window_index_buffer,
                0,
                vk::IndexType::UINT16,
            );
            device.cmd_draw_indexed(
                self.current_command_buffer.command_buffer,
                self.companion_window_index_size / 2,
                1,
                0,
                0,
                0,
            );
        }

        // Draw right eye texture to companion window
        unsafe {
            device.cmd_bind_descriptor_sets(
                self.current_command_buffer.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[DESCRIPTOR_SET_COMPANION_RIGHT_TEXTURE]],
                &[],
            );

            device.cmd_draw_indexed(
                self.current_command_buffer.command_buffer,
                self.companion_window_index_size / 2,
                1,
                self.companion_window_index_size / 2,
                0,
                0,
            );
        }

        // End the renderpass
        unsafe {
            device.cmd_end_render_pass(self.current_command_buffer.command_buffer);
        }

        // Transition the swapchain image to PRESENT_SRC for presentation
        image_memory_barrier[0].src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        image_memory_barrier[0].dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        image_memory_barrier[0].old_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        image_memory_barrier[0].new_layout = vk::ImageLayout::PRESENT_SRC_KHR;
        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }

        // Transition both of the eye textures to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL for SteamVR which requires this layout for submit
        image_memory_barrier[0].image = self.left_eye_desc.image;
        image_memory_barrier[0].src_access_mask = vk::AccessFlags::SHADER_READ;
        image_memory_barrier[0].dst_access_mask = vk::AccessFlags::TRANSFER_READ;
        image_memory_barrier[0].old_layout = self.left_eye_desc.image_layout;
        image_memory_barrier[0].new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }
        self.left_eye_desc.image_layout = image_memory_barrier[0].new_layout;

        image_memory_barrier[0].image = self.right_eye_desc.image;

        unsafe {
            device.cmd_pipeline_barrier(
                self.current_command_buffer.command_buffer,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }
        self.right_eye_desc.image_layout = image_memory_barrier[0].new_layout;

        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Gets a Matrix Projection Eye with respect to nEye.
    //-----------------------------------------------------------------------------
    fn get_hmd_matrix_projection_eye(&self, eye: openvr::Eye) -> Matrix4<f32> {
        if let Some(hmd) = &self.hmd {
            let mat = hmd.projection_matrix(eye, self.near_clip, self.far_clip);
            return Self::convert_steam_vr_matrix4_to_matrix4(&mat);
        } else {
            return Matrix4::identity();
        }
    }

    //-----------------------------------------------------------------------------
    // Purpose: Gets an HMDMatrixPoseEye with respect to nEye.
    //-----------------------------------------------------------------------------
    fn get_hmd_matrix_pose_eye(&self, eye: openvr::Eye) -> Matrix4<f32> {
        if let Some(hmd) = &self.hmd {
            let mat = hmd.eye_to_head_transform(eye);
            return Self::convert_steam_vr_matrix3_to_matrix4(&mat)
                .try_inverse()
                .expect("Eye pose is not a valid pose");
        } else {
            return Matrix4::identity();
        }
    }

    //-----------------------------------------------------------------------------
    // Purpose: Gets a Current View Projection Matrix with respect to nEye,
    //          which may be an Eye_Left or an Eye_Right.
    //-----------------------------------------------------------------------------
    fn get_current_view_projection_matrix(&self, eye: openvr::Eye) -> Matrix4<f32> {
        match eye {
            openvr::Eye::Left => self.projection_left * self.eye_pos_left * self.hmd_pose,
            openvr::Eye::Right => self.projection_right * self.eye_pos_right * self.hmd_pose,
        }
    }

    fn update_hmd_matrix_pose(&mut self) -> Result<(), Error> {
        if let Some(hmd) = &self.hmd {
            let compositor_table: *const openvr_sys::VR_IVRCompositor_FnTable =
                openvr_sys_load(openvr_sys::IVRCompositor_Version).unwrap();
            let mut render_poses = [openvr_sys::TrackedDevicePose_t {
                mDeviceToAbsoluteTracking: openvr_sys::HmdMatrix34_t { m: [[0.; 4]; 3] },
                vVelocity: openvr_sys::HmdVector3_t { v: [0.; 3] },
                vAngularVelocity: openvr_sys::HmdVector3_t { v: [0.; 3] },
                eTrackingResult: 0,
                bPoseIsValid: false,
                bDeviceIsConnected: false,
            }; openvr::MAX_TRACKED_DEVICE_COUNT];
            let mut game_poses = render_poses.clone();
            let result = unsafe {
                (*compositor_table)
                    .WaitGetPoses
                    .map(|f| {
                        f(
                            render_poses.as_mut().as_mut_ptr() as *mut _,
                            render_poses.len() as u32,
                            game_poses.as_mut().as_mut_ptr() as *mut _,
                            game_poses.len() as u32,
                        )
                    })
                    .unwrap_or(openvr_sys::EVRCompositorError_VRCompositorError_RequestFailed)
            };
            if result != openvr_sys::EVRInitError_VRInitError_None {
                return Err(format_err!("Could not WaitGetPoses: {:?}", result));
            }
            for device in 0..openvr::MAX_TRACKED_DEVICE_COUNT {
                self.tracked_device_pose[device] = render_poses[device].into();
            }
            self.valid_pose_count = 0;
            self.pose_classes = String::with_capacity(openvr::MAX_TRACKED_DEVICE_COUNT);

            for device in 0..openvr::MAX_TRACKED_DEVICE_COUNT {
                if self.tracked_device_pose[device].pose_is_valid() {
                    self.valid_pose_count += 1;
                    self.device_pose[device] = Self::convert_steam_vr_matrix3_to_matrix4(
                        self.tracked_device_pose[device].device_to_absolute_tracking(),
                    );
                    if self.dev_class_char[device] == '\0' {
                        self.dev_class_char[device] = match hmd.tracked_device_class(device as u32)
                        {
                            openvr::TrackedDeviceClass::Controller => 'C',
                            openvr::TrackedDeviceClass::HMD => 'H',
                            openvr::TrackedDeviceClass::Invalid => 'I',
                            openvr::TrackedDeviceClass::GenericTracker => 'G',
                            openvr::TrackedDeviceClass::TrackingReference => 'T',
                            _ => '?',
                        };
                    }
                    self.pose_classes.push(self.dev_class_char[device]);
                }
            }

            if self.tracked_device_pose[openvr::tracked_device_index::HMD as usize].pose_is_valid()
            {
                self.hmd_pose = self.device_pose[openvr::tracked_device_index::HMD as usize]
                    .try_inverse()
                    .unwrap();
            }
        }
        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Finds a render model we've already loaded or loads a new one
    // XXX: Instead of returning a pointer to the render model directly, an index into
    // self.render_models_vec is returned.
    //-----------------------------------------------------------------------------
    fn find_or_load_render_model(
        &mut self,
        tracked_device_index: openvr::TrackedDeviceIndex,
        render_model_name: String,
    ) -> Result<usize, Error> {
        // To simplify the Vulkan rendering code, create an instance of the model for each model name.  This is less efficient
        // memory wise, but simplifies the rendering code so we can store the transform in a constant buffer associated with
        // the model itself.  You would not want to do this in a production application.
        // XXX: This code is also commented out in the original.
        //for model in &self.render_models_vec {
        //if model.get_name() == render_model_name.to_string_lossy() {
        //return Ok((model));
        //}
        //}
        // XXX: Init a new openvr::RenderModels, so we can call methods on &mut self
        let vr_render_models = self.ovr_context.as_ref().unwrap().render_models()?;
        let model;
        let texture;
        {
            // load the model if we didn't find one
            loop {
                if let Some(m) = vr_render_models
                    .load_render_model(&CString::new(render_model_name.as_str()).unwrap())?
                {
                    model = m;
                    break;
                } else {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            loop {
                if let Some(tex) = vr_render_models.load_texture(
                    model
                        .diffuse_texture_id()
                        .expect("All render models must have textures"),
                )? {
                    texture = tex;
                    break;
                } else {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
        let mut render_model = VulkanRenderModel::new(render_model_name);
        let descriptor_sets = [
            self.descriptor_sets
                [DESCRIPTOR_SET_LEFT_EYE_RENDER_MODEL0 + tracked_device_index as usize],
            self.descriptor_sets
                [DESCRIPTOR_SET_RIGHT_EYE_RENDER_MODEL0 + tracked_device_index as usize],
        ];
        // If this gets called during HandleInput() there will be no command buffer current, so create one
        // and submit it immediately.
        let mut new_command_buffer = false;
        if self.current_command_buffer.command_buffer == vk::CommandBuffer::null() {
            self.current_command_buffer = self.get_command_buffer()?;
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                self.device.as_ref().unwrap().begin_command_buffer(
                    self.current_command_buffer.command_buffer.clone(),
                    &command_buffer_begin_info,
                )
            }?;
            new_command_buffer = true;
        }
        let device = self.device.as_ref().unwrap();
        render_model.init(
            device,
            self.physical_device_memory_properties,
            self.current_command_buffer.command_buffer,
            tracked_device_index,
            &descriptor_sets,
            model,
            texture,
        )?;
        self.render_models_vec.push(render_model);

        // If this is during HandleInput() there is was no command buffer current, so submit it now.
        if new_command_buffer {
            unsafe {
                device.end_command_buffer(self.current_command_buffer.command_buffer.clone())
            }?;

            let command_buffers = [self.current_command_buffer.command_buffer.clone()];
            let submit_info = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build()];
            unsafe {
                device.queue_submit(self.queue, &submit_info, self.current_command_buffer.fence)
            }?;
            self.current_command_buffer.command_buffer = vk::CommandBuffer::null();
            self.current_command_buffer.fence = vk::Fence::null();
        }
        return Ok(self.render_models_vec.len() - 1);
    }

    //-----------------------------------------------------------------------------
    // Purpose: Create/destroy Vulkan a Render Model for a single tracked device
    //-----------------------------------------------------------------------------
    fn setup_render_model_for_tracked_device(
        &mut self,
        tracked_device_index: openvr::TrackedDeviceIndex,
    ) -> Result<(), Error> {
        if tracked_device_index >= openvr::MAX_TRACKED_DEVICE_COUNT as u32 {
            // XXX: Does this case actually happen?
            return Ok(());
        }

        // try to find a model we've already set up
        let render_model_name = get_tracked_device_string(
            self.hmd.as_ref().unwrap(),
            tracked_device_index,
            openvr::property::RenderModelName_String,
        )?;
        let render_model_index =
            self.find_or_load_render_model(tracked_device_index, render_model_name)?;
        self.tracked_device_to_render_model[tracked_device_index as usize] =
            Some(render_model_index);
        self.show_tracked_device[tracked_device_index as usize] = true;
        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Create/destroy Vulkan Render Models
    //-----------------------------------------------------------------------------
    fn setup_render_models(&mut self) -> Result<(), Error> {
        self.tracked_device_to_render_model = [None; openvr::MAX_TRACKED_DEVICE_COUNT];

        if self.hmd.is_none() {
            return Ok(());
        }

        for tracked_device in
            (openvr::tracked_device_index::HMD + 1)..(openvr::MAX_TRACKED_DEVICE_COUNT as u32)
        {
            if !self
                .hmd
                .as_ref()
                .unwrap()
                .is_tracked_device_connected(tracked_device)
            {
                continue;
            }
            if let Err(err) = self.setup_render_model_for_tracked_device(tracked_device) {
                dprintln!(
                    "Error setting up model for tracked device {}: {}",
                    tracked_device,
                    err
                );
            }
        }
        return Ok(());
    }

    fn convert_steam_vr_matrix4_to_matrix4(mat: &[[f32; 4]; 4]) -> Matrix4<f32> {
        // XXX: Equivalent to: Matrix4::from(*mat).transpose()
        Matrix4::new(
            mat[0][0], mat[0][1], mat[0][2], mat[0][3], // Row 0
            mat[1][0], mat[1][1], mat[1][2], mat[1][3], // Row 1
            mat[2][0], mat[2][1], mat[2][2], mat[2][3], // Row 2
            mat[3][0], mat[3][1], mat[3][2], mat[3][3], // Row 3
        )
    }

    fn convert_steam_vr_matrix3_to_matrix4(mat: &[[f32; 4]; 3]) -> Matrix4<f32> {
        Matrix4::new(
            mat[0][0], mat[0][1], mat[0][2], mat[0][3], // Row 0
            mat[1][0], mat[1][1], mat[1][2], mat[1][3], // Row 1
            mat[2][0], mat[2][1], mat[2][2], mat[2][3], // Row 2
            0., 0., 0., 1., // Row 3
        )
    }
}

impl VulkanRenderModel {
    fn new(model_name: String) -> Self {
        VulkanRenderModel {
            device: None,
            physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties::default(),
            vertex_buffer: vk::Buffer::default(),
            vertex_buffer_memory: vk::DeviceMemory::default(),
            index_buffer: vk::Buffer::default(),
            index_buffer_memory: vk::DeviceMemory::default(),
            image: vk::Image::default(),
            image_memory: vk::DeviceMemory::default(),
            image_view: vk::ImageView::default(),
            image_staging_buffer: vk::Buffer::default(),
            image_staging_buffer_memory: vk::DeviceMemory::default(),
            constant_buffer: [vk::Buffer::default(); 2],
            constant_buffer_memory: [vk::DeviceMemory::default(); 2],
            constant_buffer_data: [ptr::null_mut(); 2],
            descriptor_sets: [vk::DescriptorSet::default(); 2],
            sampler: vk::Sampler::default(),
            vertex_count: usize::default(),
            tracked_device_index: openvr::TrackedDeviceIndex::default(),
            _model_name: model_name,
        }
    }

    //-----------------------------------------------------------------------------
    // Purpose: Allocates and populates the Vulkan resources for a render model
    //-----------------------------------------------------------------------------
    fn init(
        &mut self,
        device: &ash::Device,
        physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
        command_buffer: vk::CommandBuffer,
        tracked_device_index: u32,
        descriptor_sets: &[vk::DescriptorSet],
        vr_model: openvr::render_models::Model,
        vr_diffuse_texture: openvr::render_models::Texture,
    ) -> Result<(), Error> {
        self.device = Some(device.clone());
        self.physical_device_memory_properties = physical_device_memory_properties;
        self.tracked_device_index = tracked_device_index;
        self.descriptor_sets[0] = descriptor_sets[0];
        self.descriptor_sets[1] = descriptor_sets[1];

        // Create and populate the vertex buffer

        let vulkan_buffer = create_vulkan_buffer(
            device,
            &self.physical_device_memory_properties,
            Some(&vr_model.vertices()),
            (mem::size_of::<openvr::render_models::Vertex>() * vr_model.vertices().len()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        self.vertex_buffer = vulkan_buffer.0;
        self.vertex_buffer_memory = vulkan_buffer.1;

        // Create and populate the index buffer
        let vulkan_buffer = create_vulkan_buffer(
            device,
            &self.physical_device_memory_properties,
            Some(&vr_model.indices()),
            (mem::size_of::<u16>() * vr_model.indices().len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        self.index_buffer = vulkan_buffer.0;
        self.index_buffer_memory = vulkan_buffer.1;

        // create and populate the texture
        {
            let (width, height) = vr_diffuse_texture.dimensions();
            let image_width = width as u32;
            let image_height = height as u32;
            let channels: u32 = 4;

            // Copy the base level to a buffer, reserve space for mips (overreserve by a bit to avoid having to calc mipchain size ahead of time)
            let mut buffer = vec![0; (image_width * image_height * channels * 2) as usize];
            buffer[..(image_width * image_height * channels) as usize]
                .copy_from_slice(vr_diffuse_texture.data());

            // XXX: pointers replaced with indices so we can use split_at_mut
            let mut prev_start = 0;
            let mut cur_start = (image_width * image_height * channels) as usize;

            let mut buffer_image_copies = Vec::new();

            let mut buffer_image_copy = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    base_array_layer: 0,
                    layer_count: 1,
                    mip_level: 0,
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                },
            };
            buffer_image_copies.push(buffer_image_copy.clone());

            let mut mip_width = image_width;
            let mut mip_height = image_height;

            while mip_width > 1 && mip_height > 1 {
                let (src, dst) = buffer.split_at_mut(cur_start as usize);
                let src = &mut src[prev_start..];

                let (new_mip_width, new_mip_height) =
                    MainApplication::gen_mip_map_rgba(src, dst, mip_width, mip_height);
                mip_width = new_mip_width;
                mip_height = new_mip_height;
                buffer_image_copy.buffer_offset = cur_start as u64;
                buffer_image_copy.image_subresource.mip_level += 1;
                buffer_image_copy.image_extent.width = mip_width;
                buffer_image_copy.image_extent.height = mip_height;
                buffer_image_copies.push(buffer_image_copy.clone());
                prev_start = cur_start;
                cur_start += (mip_width * mip_height * channels) as usize;
            }

            let buffer_size: vk::DeviceSize = cur_start as u64;

            // Create the image
            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                })
                .mip_levels(buffer_image_copies.len() as u32)
                .array_layers(1)
                .format(vk::Format::R8G8B8A8_UNORM)
                .tiling(vk::ImageTiling::OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .flags(vk::ImageCreateFlags::empty())
                .build();

            self.image = unsafe { device.create_image(&image_create_info, None) }?;

            let memory_requirements = unsafe { device.get_image_memory_requirements(self.image) };

            let memory_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(memory_requirements.size)
                .memory_type_index(
                    memory_type_from_properties(
                        &self.physical_device_memory_properties,
                        memory_requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                    .ok_or(err_msg("Could not get memory type index"))?,
                );

            self.image_memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }?;
            unsafe { device.bind_image_memory(self.image, self.image_memory, 0) }?;

            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .flags(vk::ImageViewCreateFlags::empty())
                .image(self.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image_create_info.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: image_create_info.mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.image_view = unsafe { device.create_image_view(&image_view_create_info, None) }?;

            // Create a staging buffer
            let vulkan_buffer = create_vulkan_buffer(
                device,
                &self.physical_device_memory_properties,
                Some(&buffer),
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            self.image_staging_buffer = vulkan_buffer.0;
            self.image_staging_buffer_memory = vulkan_buffer.1;

            // Copy memory to the staging buffer
            unsafe {
                let data = device.map_memory(
                    self.image_staging_buffer_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::default(),
                )?;
                data.copy_from_nonoverlapping(
                    buffer.as_ptr() as *const c_void,
                    buffer_size as usize,
                );
                // XXX: Spec says we need to unmap *after* flushing.
            }
            let memory_range = vk::MappedMemoryRange::builder()
                .memory(self.image_staging_buffer_memory)
                .size(vk::WHOLE_SIZE)
                .build();
            unsafe {
                device.flush_mapped_memory_ranges(&[memory_range])?;
                device.unmap_memory(self.image_staging_buffer_memory);
            }

            // Transition the image to TRANSFER_DST to receive image
            let mut image_memory_barrier = [vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: image_create_info.mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build()];

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_memory_barrier,
                );
            }

            // Issue the copy to fill the image data
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.image_staging_buffer,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &buffer_image_copies,
                );
            }

            // Transition the image to SHADER_READ_OPTIMAL for reading
            image_memory_barrier[0].src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_memory_barrier[0].dst_access_mask = vk::AccessFlags::SHADER_READ;
            image_memory_barrier[0].old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_memory_barrier[0].new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_memory_barrier,
                );
            }

            // Create the sampler
            let sampler_create_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .min_lod(0.0)
                .max_lod(image_create_info.mip_levels as f32);

            self.sampler = unsafe { device.create_sampler(&sampler_create_info, None) }?;
        }

        // Create a constant buffer to hold the transform (one for each eye)
        for eye in 0..2 {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(mem::size_of::<Matrix4<f32>>() as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);
            self.constant_buffer[eye] = unsafe { device.create_buffer(&buffer_create_info, None) }?;

            let memory_requirements =
                unsafe { device.get_buffer_memory_requirements(self.constant_buffer[eye]) };
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .memory_type_index(
                    memory_type_from_properties(
                        &physical_device_memory_properties,
                        memory_requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE,
                    )
                    .expect("Failed to find matching memory type index for buffer"),
                )
                .allocation_size(memory_requirements.size);

            self.constant_buffer_memory[eye] =
                unsafe { device.allocate_memory(&alloc_info, None)? };
            unsafe {
                device.bind_buffer_memory(
                    self.constant_buffer[eye],
                    self.constant_buffer_memory[eye],
                    0,
                )?;
            }

            // Map and keep mapped persistently
            self.constant_buffer_data[eye] = unsafe {
                device.map_memory(
                    self.constant_buffer_memory[eye],
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            }? as *mut Matrix4<f32>;

            // Bake the descriptor set
            let buffer_info = [vk::DescriptorBufferInfo::builder()
                .buffer(self.constant_buffer[eye].clone())
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()];

            let image_info = [vk::DescriptorImageInfo::builder()
                .image_view(self.image_view.clone())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()];
            let sampler_info = [vk::DescriptorImageInfo::builder()
                .sampler(self.sampler.clone())
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[eye])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[eye])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[eye])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_info)
                    .build(),
            ];
            unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };
        }
        self.vertex_count = vr_model.indices().len();

        return Ok(());
    }

    //-----------------------------------------------------------------------------
    // Purpose: Frees the Vulkan resources for a render model
    //-----------------------------------------------------------------------------
    fn cleanup(&mut self) {
        unsafe {
            let device = self.device.as_ref().unwrap();
            if self.vertex_buffer != vk::Buffer::null() {
                device.destroy_buffer(self.vertex_buffer, None);
                self.vertex_buffer = vk::Buffer::null();
            }

            if self.vertex_buffer_memory != vk::DeviceMemory::null() {
                device.free_memory(self.vertex_buffer_memory, None);
                self.vertex_buffer_memory = vk::DeviceMemory::null();
            }

            if self.index_buffer != vk::Buffer::null() {
                device.destroy_buffer(self.index_buffer, None);
                self.index_buffer = vk::Buffer::null();
            }

            if self.index_buffer_memory != vk::DeviceMemory::null() {
                device.free_memory(self.index_buffer_memory, None);
                self.index_buffer_memory = vk::DeviceMemory::null();
            }

            if self.image != vk::Image::null() {
                device.destroy_image(self.image, None);
                self.image = vk::Image::null();
            }

            if self.image_memory != vk::DeviceMemory::null() {
                device.free_memory(self.image_memory, None);
                self.image_memory = vk::DeviceMemory::null();
            }

            if self.image_view != vk::ImageView::null() {
                device.destroy_image_view(self.image_view, None);
                self.image_view = vk::ImageView::null();
            }

            if self.image_staging_buffer != vk::Buffer::null() {
                device.destroy_buffer(self.image_staging_buffer, None);
                self.image_staging_buffer = vk::Buffer::null();
            }

            for eye in 0..2 {
                if self.constant_buffer[eye] != vk::Buffer::null() {
                    device.destroy_buffer(self.constant_buffer[eye], None);
                    self.constant_buffer[eye] = vk::Buffer::null();
                }
                if self.constant_buffer_memory[eye] != vk::DeviceMemory::null() {
                    device.free_memory(self.constant_buffer_memory[eye], None);
                    self.constant_buffer_memory[eye] = vk::DeviceMemory::null();
                }
            }

            if self.sampler == vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }
        }
    }

    fn draw(
        &self,
        eye: openvr::Eye,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        mat: &Matrix4<f32>,
    ) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        unsafe {
            // Update the CB with the transform
            *self.constant_buffer_data[eye as usize] = *mat;

            // Bind the descriptor set
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[self.descriptor_sets[eye as usize]],
                &[],
            );

            // Bind the VB/IB and draw
            let offsets = [0];

            device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &offsets);

            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );
            device.cmd_draw_indexed(command_buffer, self.vertex_count as u32, 1, 0, 0, 0);
        }
        return Ok(());
    }
}

impl Drop for VulkanRenderModel {
    fn drop(&mut self) {
        self.cleanup();
    }
}

fn main() -> Result<(), Error> {
    let mut main_application = MainApplication::new(std::env::args());

    let result = if let Err(err) = main_application.init() {
        Err(err)
    } else {
        main_application.run_main_loop()
    };
    if let Err(e) = main_application.shutdown() {
        dprintln!("Error in shutdown: {:?}", e);
    }
    return result;
}

// XXX: Below here are utility functions which were either not present or included from headers in
// the original file.

/**
 * Convert Vulkan's fixed length "string" to std::str
 */
fn vulkan_str(char_block: &[i8; 256]) -> Result<&str, str::Utf8Error> {
    // XXX: Handle names that might not have null terminators.
    let len = char_block.iter().position(|c| *c == 0).unwrap_or(256);
    let name_bytes: &[u8] = transmute_to_bytes(&char_block[0..len]);
    return str::from_utf8(name_bytes);
}

/**
 * Convert a slice of CString to a Vec of raw pointers
 * The resulting pointers must not be used after cstrings is mutated.
 */
fn to_raw_cstr_array(cstrings: &[CString]) -> Vec<*const c_char> {
    cstrings
        .iter()
        .map(|s| s.as_ptr() as *const c_char)
        .collect()
}

/**
 * Convert a slice of CString to a Vec of raw pointers
 * The resulting pointers must not be used after cstrings is mutated.
 */
fn openvr_sys_load<T>(suffix: &[u8]) -> Result<*const T, Error> {
    let mut magic = Vec::from(b"FnTable:".as_ref());
    magic.extend(suffix);
    let mut error = openvr_sys::EVRInitError_VRInitError_None;
    let result =
        unsafe { openvr_sys::VR_GetGenericInterface(magic.as_ptr() as *const i8, &mut error) };
    if error != openvr_sys::EVRInitError_VRInitError_None {
        let msg =
            unsafe { CStr::from_ptr(openvr_sys::VR_GetVRInitErrorAsEnglishDescription(error)) };
        return Err(err_msg(msg.to_string_lossy().to_owned()));
    }
    Ok(result as *const T)
}

// XXX We can't use SDL on Windows due to bindgen issues, so we just use this winit code instead,
// taken from unknownue/vulkan-tutorial-rs
/**
 * Create an XlibSurface.
 */
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::platform::unix::WindowExtUnix;

    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR {
        s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        window: x11_window as vk::Window,
        dpy: x11_display as *mut vk::Display,
    };
    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

/**
 * Create a MaxOSSurface.
 */
#[cfg(target_os = "macos")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::mem;
    use winit::platform::macos::WindowExtMacOS;

    let wnd: cocoa_id = mem::transmute(window.ns_window());

    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_M,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.ns_view() as *const c_void,
    };

    let macos_surface_loader = MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface_mvk(&create_info, None)
}

/**
 * Create a Win32Surface.
 */
#[cfg(target_os = "windows")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::platform::windows::WindowExtWindows;

    let hwnd = window.hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}

/**
 * Open a file given a path relative to the openvr sdk samples/bin directory.
 * If the file is not present locally, download it from github and cache it in the same directory
 * as the exe.
 */
fn acquire_file(path: &Path) -> Result<fs::File, Error> {
    assert!(path.is_relative());
    let mut base = std::env::current_exe().unwrap();
    base.pop();
    let full_path = base.join(path);
    match fs::File::open(&full_path) {
        Ok(file) => Ok(file),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let mut directory = full_path.clone();
            directory.pop();
            fs::create_dir_all(&directory).unwrap();
            let mut url = String::from(
                "https://raw.githubusercontent.com/ValveSoftware/openvr/master/samples/bin/",
            );
            url.push_str(&path.to_slash().unwrap());
            let mut buffer = Vec::new();
            let resp = ureq::get(&url).call()?;
            resp.into_reader().read_to_end(&mut buffer)?;
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&full_path)
                .unwrap();
            file.write(&buffer).unwrap();
            file.seek(SeekFrom::Start(0)).unwrap();
            Ok(file)
        }
        Err(err) => Err(err.into()),
    }
}
