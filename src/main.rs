// https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-ONNXRuntime-Rust
// https://github.com/pykeio/ort/blob/main/examples/yolov8/examples/yolov8.rs

use std::env;
use std::path::Path;

use anyhow::Result;

mod model;
mod video;

const MODEL_PATH: &str = r"assets\weights\yolov8m.onnx";

fn main() -> Result<()> {
  let file_path = env::args()
    .nth(1)
    .expect("Please supply a video or image file!");
  let file_path = Path::new(file_path.as_str());

  assert!(file_path.is_file(), "Please specify a true file path!");

  ort::init()
    .with_execution_providers([ort::CUDAExecutionProvider::default().build()])
    .commit()?;
  let yolo = model::YOLOv8::new(MODEL_PATH)?;

  video::init()?;
  let mut video_file = video::format::input(&file_path)?;

  // This takes long time
  video::process_video(&mut video_file, |frame| {
    // This also takes long time
    let image = frame.to_dynamic_image()?;
    let data = yolo.process(image)?;

    println!("Data: {data:?}");

    Ok(())
  })?;

  Ok(())
}
