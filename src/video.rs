extern crate ffmpeg_next as ffmpeg;
pub use ffmpeg::*;

// use itertools::Itertools;

use self::util::frame::video::Video;
use image::{DynamicImage, ImageFormat, ImageOutputFormat, ImageResult, RgbImage};
use std::io::Cursor;

pub struct Frame<'a> {
  width: u32,
  height: u32,
  data: &'a [u8],
}

impl<'a> Default for Frame<'a> {
  fn default() -> Self {
    Self {
      data: &[],
      width: 0,
      height: 0,
    }
  }
}

impl<'a> Frame<'a> {
  pub fn new(width: u32, height: u32, data: &'a [u8]) -> Self {
    Frame {
      width,
      height,
      data,
    }
  }

  pub fn from_video(video: &'a Video) -> Self {
    Self::new(video.width(), video.height(), video.data(0))
  }

  pub fn to_buffer(&self) -> Option<RgbImage> {
    RgbImage::from_vec(self.width, self.height, self.data.to_vec())
  }

  pub fn to_png_buffer(&self) -> ImageResult<Cursor<Vec<u8>>> {
    let mut png_buffer = Cursor::new(Vec::new());
    let buffer = self.to_buffer().unwrap();

    buffer.write_to(&mut png_buffer, ImageOutputFormat::Png)?;
    png_buffer.set_position(0);

    Ok(png_buffer)
  }

  pub fn to_dynamic_image(&self) -> ImageResult<DynamicImage> {
    let cursor = self.to_png_buffer()?;

    image::load(cursor, ImageFormat::Png)
  }
}

// Finally this fucking works!
/// Converts [`Video`](Video)'s first frame to [`DynamicImage`](DynamicImage)
// pub fn convert_dynamic_image(video: Video, format: ImageFormat) -> ImageResult<DynamicImage> {
//   let cursor = io::Cursor::new(video.data(0));

//   image::load(cursor, format)
// }

pub fn process_video(
  input: &mut format::context::Input,
  process: impl Fn(Frame) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
  let video_stream = input
    .streams()
    .best(media::Type::Video)
    .ok_or(Error::StreamNotFound)?;
  let video_stream_index = video_stream.index();

  let context_decoder = codec::context::Context::from_parameters(video_stream.parameters())?;
  let mut decoder = context_decoder.decoder().video()?;

  let mut scaler = software::scaling::Context::get(
    decoder.format(),
    decoder.width(),
    decoder.height(),
    format::Pixel::RGB24,
    decoder.width(),
    decoder.height(),
    software::scaling::flag::Flags::BILINEAR,
  )?;
  let mut frame_index = 0;

  let mut receive_and_process_decoded_frames =
    |decoder: &mut ffmpeg::decoder::Video| -> anyhow::Result<()> {
      let mut decoded = Video::empty();

      while decoder.receive_frame(&mut decoded).is_ok() {
        let mut rgb_video = Video::empty();
        scaler.run(&decoded, &mut rgb_video)?;
        let frame = Frame::from_video(&rgb_video);

        // let rgb_pixels = frame.data.iter().tuples::<(_, _, _)>().map(|(r, g, b)| {
        //   image::Pixel::from_slice(slice)
        // });
        // let buffer = frame.to_buffer();

        process(frame)?;

        frame_index += 1;
      }

      // println!("Frame count: {frame_index}");

      Ok(())
    };

  for (stream, packet) in input.packets() {
    if stream.index() == video_stream_index {
      decoder.send_packet(&packet)?;
      receive_and_process_decoded_frames(&mut decoder)?;
    }
  }
  decoder.send_eof()?;
  receive_and_process_decoded_frames(&mut decoder)?;

  Ok(())
}

// pub fn slice_to_frames<'a>(video: &'a Video) -> Vec<Frame<'a>> {
//   let metadata = video.metadata();
//   let frame_count = metadata.get("frames");

//   let mut frames: Vec<Frame> = Vec::new();

//   if let Some(frame_count) = frame_count {
//     for index in 0..frame_count.parse().unwrap() {
//       frames.push(Frame::new(video.data(index)));
//     }
//   }

//   frames
// }
