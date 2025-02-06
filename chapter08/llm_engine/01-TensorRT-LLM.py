import tempfile
from tensorrt_llm import LLM, SamplingParams

def main():
    # 模型可以接受 Hugging Face 模型名称、本地 Hugging Face 模型的路径，
    # 或者 TensorRT Model Optimizer 的量化检查点，例如 Hugging Face 上的 nvidia/Llama-3.1-8B-Instruct-FP8。
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 你可以将引擎保存到磁盘，稍后再加载。LLM 类可以接受 Hugging Face 模型或 TRT-LLM 引擎。
    llm.save(tempfile.mkdtemp())

    # 示例提示词。
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # 创建采样参数。
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 对每个提示词生成文本。
    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    # 输出示例：
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The president of the United States is', Generated text: 'likely to nominate a new Supreme Court justice to fill the seat vacated by the death of Antonin Scalia. The Senate should vote to confirm the'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'

if __name__ == '__main__':
    main()