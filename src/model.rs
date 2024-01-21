pub use ort::Result;
use ort::{inputs, Session, SessionOutputs};

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};

use ndarray::{s, Array, Axis, Dim};

pub const YOLOV8_WIDTH: u32 = 640;
pub const YOLOV8_HEIGHT: u32 = 640;
#[rustfmt::skip]
pub const YOLOV8_CLASS_LABELS: [&str; 80] = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
  pub x1: f32,
  pub y1: f32,
  pub x2: f32,
  pub y2: f32,
}

pub fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
  (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
  ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
    - intersection(box1, box2)
}

pub struct YOLOv8 {
  model: Session,
}

impl YOLOv8 {
  pub fn new(model_path: &str) -> Result<Self> {
    Ok(Self {
      model: Session::builder()?.with_model_from_file(model_path)?,
    })
  }

  fn process_image(&self, img: DynamicImage) -> Array<f32, Dim<[usize; 4]>> {
    // This takes long time
    let resized_img = img.resize_exact(YOLOV8_WIDTH, YOLOV8_HEIGHT, FilterType::CatmullRom);

    let mut input = Array::zeros((1, 3, 640, 640));
    for pixel in resized_img.pixels() {
      let x = pixel.0 as _;
      let y = pixel.1 as _;
      let [r, g, b, _] = pixel.2 .0;
      input[[0, 0, y, x]] = (r as f32) / 255.;
      input[[0, 1, y, x]] = (g as f32) / 255.;
      input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    input
  }

  pub fn process(&self, img: image::DynamicImage) -> Result<Vec<(BoundingBox, &str, f32)>> {
    let (img_width, img_height) = (img.width(), img.height());
    let input = self.process_image(img);

    let outputs: SessionOutputs = self.model.run(inputs!["images" => input.view()]?)?;
    let output = outputs["output0"]
      .extract_tensor::<f32>()
      .unwrap()
      .view()
      .t()
      .into_owned();

    let mut boxes: Vec<(BoundingBox, &str, f32)> = Vec::new();
    let output = output.slice(s![.., .., 0]);

    for row in output.axis_iter(Axis(0)) {
      let row: Vec<_> = row.iter().copied().collect();
      let (class_id, prob) = row
        .iter()
        // skip bounding box coordinates
        .skip(4)
        .enumerate()
        .map(|(index, value)| (index, *value))
        .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
        .unwrap();

      if prob < 0.5 {
        continue;
      }

      let label = YOLOV8_CLASS_LABELS[class_id];
      let xc = row[0] / 640. * (img_width as f32);
      let yc = row[1] / 640. * (img_height as f32);
      let w = row[2] / 640. * (img_width as f32);
      let h = row[3] / 640. * (img_height as f32);

      boxes.push((
        BoundingBox {
          x1: xc - w / 2.,
          y1: yc - h / 2.,
          x2: xc + w / 2.,
          y2: yc + h / 2.,
        },
        label,
        prob,
      ));
    }

    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    let mut result: Vec<(BoundingBox, &str, f32)> = Vec::new();

    while !boxes.is_empty() {
      result.push(boxes[0]);
      boxes = boxes
        .iter()
        .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
        .copied()
        .collect();
    }

    Ok(result)
  }
}
