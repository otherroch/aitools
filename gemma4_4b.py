from transformers import AutoProcessor, AutoModelForMultimodalLM
import torch


# https://huggingface.co/google/gemma-4-31B-it
# https://huggingface.co/google/gemma-4-E4B-it
# https://huggingface.co/google/gemma-4-26B-A4B-it
# https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4

# https://huggingface.co/docs/transformers/en/model_doc/gemma4#transformers.Gemma4VideoProcessor
# https://huggingface.co/docs/transformers/en/model_doc/gemma4#transformers.Gemma4ForMultimodalLM
# https://huggingface.co/docs/transformers/en/model_doc/gemma4#transformers.Gemma4Model

# https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gemma4.md
#MODEL_ID = "google/gemma-4-E4B-it"
#MODEL_ID = "google/gemma-4-26B-A4B-it"
MODEL_ID = "ebircak/gemma-4-31B-it-4bit-W4A16-AWQ"
#MODEL_ID = "nvidia/Gemma-4-31B-IT-NVFP4"
model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
# Prompt - add video before text
messages = [
    {
        'role': 'user',
        'content': [
            {"type": "video", "video": "/home/roch/aitools/gemma4_test/ctv30.mp4"},
            {'type': 'text', 'text': 'Describe this video.'}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
text_output = processor.parse_response(response)
print("output from gemm4: ", text_output["content"])  
