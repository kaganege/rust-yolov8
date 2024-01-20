extern crate ffmpeg_next as ffmpeg;

pub use ffmpeg::*;
use image::{DynamicImage, ImageFormat, ImageResult};
use std::io;

// Finally this fucking works!
/// Converts [`Video`](self::util::frame::video::Video)'s first frame to [`DynamicImage`](DynamicImage)
pub fn convert_dynamic_image(
  video: self::util::frame::video::Video,
  format: ImageFormat,
) -> ImageResult<DynamicImage> {
  let cursor = io::Cursor::new(video.data(0));

  image::load(cursor, format)
}
