// https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-ONNXRuntime-Rust
// https://github.com/pykeio/ort/blob/main/examples/yolov8/examples/yolov8.rs

use std::env;
use std::path::Path;

use anyhow::Result;

mod model;
mod video;

fn main() -> Result<()> {
  let file_path = env::args()
    .nth(1)
    .expect("Please supply a video or image file!");
  let file_path = Path::new(file_path.as_str());

  assert!(file_path.is_file(), "Please specify a true file path!");

  video::init()?;
  ort::init()
    .with_execution_providers([ort::CUDAExecutionProvider::default().build()])
    .commit()?;

  let video_file = video::format::input(&file_path)?;
  let video_stream = video_file
    .streams()
    .best(video::media::Type::Video)
    .ok_or(video::Error::StreamNotFound)?;
  let video_stream_index = video_stream.index();

  let context_decoder = video::codec::context::Context::from_parameters(video_stream.parameters())?;
  let mut decoder = context_decoder.decoder().video()?;

  let mut scaler = video::software::scaling::Context::get(
    decoder.format(),
    decoder.width(),
    decoder.height(),
    video::format::Pixel::RGB24,
    decoder.width(),
    decoder.height(),
    video::software::scaling::flag::Flags::BILINEAR,
  )?;

  let mut frame_index = 0;

  let mut receive_and_process_decoded_frames =
    |decoder: &mut video::decoder::Video| -> Result<(), video::Error> {
      let mut decoded = video::util::frame::video::Video::empty();
      while decoder.receive_frame(&mut decoded).is_ok() {
        let mut rgb_frame = video::util::frame::video::Video::empty();
        scaler.run(&decoded, &mut rgb_frame)?;
        let image = video::convert_dynamic_image(rgb_frame, image::ImageFormat::Png);
        // FIXME: process image

        frame_index += 1;
      }
      Ok(())
    };

  Ok(())
}
