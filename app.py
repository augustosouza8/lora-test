"""
app.py
=====
A minimal Gradio demo to verify the Spaces environment. This simple interface confirms that the model
loads correctly and provides a starting point for later interactive features.
"""

import gradio as gr
from transformers import pipeline

def model_inference(text: str) -> str:
    """
    Run inference using the T5-small model for text-to-text generation.

    Args:
        text (str): The input text prompt.

    Returns:
        str: The generated text from the model.
    """
    generator = pipeline("text2text-generation", model="t5-small")
    result = generator(text, max_length=50)
    return result[0]['generated_text']

# Using gr.Textbox directly as inputs based on the updated Gradio API.
demo = gr.Interface(
    fn=model_inference,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs="text",
    title="LoRA Fine-Tuning Demo",
    description="A simple demo to test model loading on Hugging Face Spaces."
)

if __name__ == "__main__":
    demo.launch()
