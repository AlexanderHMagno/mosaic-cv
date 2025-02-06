import gradio as gr


class GradioInterface:

    def __init__(self, process, examples):
        self.demo = gr.Interface(
            fn=process,
            inputs= [
                gr.Image(type="pil", label="Upload an Image"),
                gr.Slider(2, 256, step=2, value=8, label="Output Grid Size (e.g., 8x8)"),
                gr.Radio(choices=["Robots","Cats", "Humans"], label="Tile Type" ,value="Robots"),
                gr.Radio(choices=["Original Size","Resize to Grid Size"], label="Keep Image Size" ,value="Original Size"),
            ],
            outputs=gr.Image(type="pil", label="Processed Image with Grid Analysis"),
            title="Image Grid Analysis",
            description="Upload an image and divide it into grids to analyze color or intensity data in each cell.",
            examples=examples
        )

    def launch(self):
        self.demo.launch()