# For running inference on the TF-Hub module.

# Flask web app for image upload and detection
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import io
import time
from flask import Flask, request, send_file, render_template_string, send_from_directory

app = Flask(__name__, static_folder='static')

detector_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
print("Loading model...")
detector = hub.load(detector_url).signatures['default']
print("Model loaded.")

def draw_bounding_box_on_image(image,
                 ymin,
                 xmin,
                 ymax,
                 xmax,
                 color,
                 font,
                 thickness=4,
                 display_str_list=()):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                  ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
         (left, top)],
        width=thickness,
        fill=color)
  display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  for display_str in display_str_list[::-1]:
    bbox = font.getbbox(display_str)
    text_width, text_height = bbox[2], bbox[3]
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
            (left + text_width, text_bottom)],
             fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
          display_str,
          fill="black",
          font=font)
    text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  colors = list(ImageColor.colormap.values())
  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()
  image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      draw_bounding_box_on_image(
        image_pil,
        ymin,
        xmin,
        ymax,
        xmax,
        color,
        font,
        display_str_list=[display_str])
  return image_pil

@app.route('/')
def index():
  return send_from_directory(app.static_folder, 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
  if 'image' not in request.files:
    return 'No image uploaded', 400
  file = request.files['image']
  img = Image.open(file.stream).convert('RGB')
  img = ImageOps.fit(img, (1280, 856), Image.LANCZOS)
  img_np = np.array(img)
  img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)[tf.newaxis, ...] / 255.0
  start_time = time.time()
  result = detector(img_tensor)
  end_time = time.time()
  print(f"Inference time: {end_time - start_time:.2f}s")
  result = {key:value.numpy() for key,value in result.items()}
  boxes = result['detection_boxes']
  class_names = result['detection_class_entities']
  scores = result['detection_scores']
  img_with_boxes = draw_boxes(img_np, boxes, class_names, scores)
  buf = io.BytesIO()
  img_with_boxes.save(buf, format='JPEG')
  buf.seek(0)
  return send_file(buf, mimetype='image/jpeg')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)