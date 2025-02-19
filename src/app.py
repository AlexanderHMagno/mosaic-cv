import gradio as gr
from const import examples
from imageProcessor import ImageProcessor

# Create an instance of the ImageProcessor class
processor = ImageProcessor()

# Create Gradio interface
demo = gr.Interface(
    fn=processor.process_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Slider(8, 32, step=4, value=8, label="Output Tile Size (e.g., 16x16)"),
        gr.Radio(choices=["Robots","Cats", "Humans"], label="Tile Type" ,value="Robots"),         
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
