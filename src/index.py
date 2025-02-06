import gradio as gr
from const import examples
from mosaic import ImageProcessor

# Create an instance of the ImageProcessor class
processor = ImageProcessor()

# Create Gradio interface
demo = gr.Interface(
    fn=processor.process_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Slider(32, 128, step=8, value=32, label="Output Grid Size (e.g., 32x32)"),
        gr.Radio(choices=["Robots","Cats", "Humans"], label="Tile Type" ,value="Robots"),
        gr.Radio(choices=["Original Size","Resize to Grid Size"], label="Keep Image Size" ,value="Original Size"),
        
    ],
    outputs=[
        gr.Image(type="pil", label="Processed Image with Grid Analysis"), 
        gr.Textbox(label="Metrics Explanation")
    ],
    title="Image Grid Analysis",
    description="Upload an image and divide it into grids to analyze color or intensity data in each cell.",
    examples=examples
)

# Run the app
if __name__ == "__main__":
    demo.launch()
