"""
app.py
======
A minimal Gradio demo to verify the Spaces environment. This simple interface confirms that the model
loads correctly and provides a starting point for later interactive features..
"""

import gradio as gr
from transformers import pipeline

def model_inference(text: str) -> str:
    # For now, use a simple text generation pipeline.
    generator = pipeline("text2text-generation", model="t5-small")
    result = generator(text, max_length=50)
    return result[0]['generated_text']

demo = gr.Interface(
    fn=model_inference,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs="text",
    title="LoRA Fine-Tuning Demo",
    description="A simple demo to test the model loading on Hugging Face Spaces."
)

if __name__ == "__main__":
    demo.launch()
