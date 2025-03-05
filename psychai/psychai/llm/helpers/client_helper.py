# 📌 标准库（Python 内置库）
import os      # 处理文件路径、环境变量等
import re      # 正则表达式，用于文本处理
import json    # 处理 JSON 数据（序列化/反序列化）
import time    # 处理时间（如延迟、时间戳）

# 📌 机器学习 & 深度学习库
import torch  # PyTorch，支持 GPU/CPU 计算

# 📌 云端 API SDK
import boto3   # AWS SDK，用于调用 AWS Bedrock API
import openai  # OpenAI SDK，用于调用 OpenAI API
import pandas as pd  # Pandas 库，用于处理数据表格（DataFrame）

# 📌 AI 供应商 & 模型管理
from enum import Enum  # 枚举类型（Enum），用于存储 AI 供应商和模型信息

# 📌 Hugging Face `transformers` 库（本地 & Hugging Face 模型）
from transformers import AutoModelForCausalLM, AutoTokenizer  # NLP 预训练模型

# 📌 知乎 AI SDK
from zhipuai import ZhipuAI  # 用于调用 知乎 AI API

# 📌 字节跳动方舟（VolcEngine Ark）SDK
# 需要先安装：`pip install volcengine-python-sdk[ark]`
from volcenginesdkarkruntime import Ark  # 处理方舟 API（字节跳动）

# class model_helper:
#     def __init__(self):
#         pass


# 📌 定义 AI 供应商（Provider）
class LLMSource(Enum):
    """AI 供应商枚举"""
    OPENAI = "openai"  # OpenAI API
    DEEPSEEK = "deepseek"  # DeepSeek API
    QWEN = "qwen"  # 阿里 API
    ZHIPUAI = "zhipuai" 
    VOLCANO = "volcano" # 豆包
    AWS = "aws"  # AWS Bedrock API
    HUGGINGFACE = "huggingface"

# 📌 定义 AI 模型（Model）
class LLMModel(Enum):
    
    """AI 模型枚举，包括 OpenAI、DeepSeek 和 AWS"""
    # OpenAI 模型
    GPT_4O = ("gpt-4o", LLMSource.OPENAI)
    GPT_4O_MINI = ("gpt-4o-mini", LLMSource.OPENAI)
    GPT_35_TURBO = ("gpt-3.5-turbo", LLMSource.OPENAI)

    # QWEN 模型
    QWEN_TURBO = ("qwen-turbo", LLMSource.QWEN)
    QWEN_PLUS = ("qwen-plus", LLMSource.QWEN)
    QWEN_MAX= ("qwen-max", LLMSource.QWEN)
    
    # DeepSeek 模型
    DEEPSEEK_CHAT = ("deepseek-chat", LLMSource.DEEPSEEK)
    DEEPSEEK_REASONER = ("deepseek-reasoner", LLMSource.DEEPSEEK)
    DEEPSEEK_CODER = ("deepseek-coder", LLMSource.DEEPSEEK)
    
    
    # AWS Bedrock 模型
    # ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    # ref: https://github.com/langchain-ai/langchain-aws/issues/258 for “on-demand throughput isn’t supported.”
    AWS_LLAMA_3_2_3B = ("us.meta.llama3-2-3b-instruct-v1:0", LLMSource.AWS)
    AWS_CLAUDE_3_5_HAIKU = ("us.anthropic.claude-3-5-haiku-20241022-v1:0", LLMSource.AWS)

    # Huggingface 模型
    HF_MS_DIALOGPT_MEDIUM =  ("microsoft/DialoGPT-medium", LLMSource.HUGGINGFACE)  

    # zhipu 模型
    ZHIPUAI_GLM_4_PLUS =  ("glm-4-plus", LLMSource.ZHIPUAI) 

    # volcano
    VOL_DOUBAO_1_5_LITE = ("doubao-1.5-lite-32k-250115", LLMSource.VOLCANO) 
    VOL_DOUBAO_1_5_PRO = ("doubao-1.5-pro-32k-250115", LLMSource.VOLCANO) 

    def __init__(self, model_name, provider):
        """
        初始化模型枚举
        :param model_name: 模型名称（如 "gpt-4o"）
        :param provider: 供应商（Source.OPENAI / Source.DEEPSEEK / Source.AWS）
        """
        self.model_name = model_name  # 存储模型名称
        self.provider = provider  # 存储供应商信息

# 📌 AI 聊天客户端
class AIChatClient:
    """
    AI 聊天客户端，支持 OpenAI、DeepSeek 和 AWS Bedrock。
    支持：
    ✅ 逐词流式输出
    ✅ 完整返回模式
    """

    def __init__(self, model: LLMModel, temperature=0.7):
        """
        初始化 AI 客户端
        :param model: 选择的 AI 模型（如 Model.GPT_4O, Model.CLAUDE_V2）
        """
        self.model = model  # 存储模型
        self.provider = model.provider  # 获取供应商（OpenAI / DeepSeek / AWS）
        self.temperature = temperature
        self.streaming_displace_delay_time = 0.05

        # 📌 根据供应商配置 API 访问
        if self.provider == LLMSource.OPENAI:
            self.api_key = os.getenv("openai_api_key")  # 请替换为您的 OpenAI API 密钥
            self.base_url = None  # OpenAI 默认 API 端点
            self.client = openai.OpenAI(api_key=self.api_key)

        if self.provider == LLMSource.QWEN:
            self.api_key = os.getenv("qwen_api_key")  # 请替换为您的 OpenAI API 密钥
            self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # OpenAI 默认 API 端点
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        elif self.provider == LLMSource.DEEPSEEK:
            self.api_key = os.getenv("deep_seek_api_key")  # 请替换为您的 DeepSeek API 密钥
            self.base_url = "https://api.deepseek.com"  # DeepSeek API 端点
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        elif self.provider == LLMSource.ZHIPUAI:
            self.api_key = os.getenv("zhipuai_api_key")  # 请替换为您的 DeepSeek API 密钥
            self.base_url = ""  # DeepSeek API 端点
            self.client = ZhipuAI(api_key=self.api_key)
           
        elif self.provider == LLMSource.VOLCANO:
            self.api_key = os.getenv("volcano_api_key")  # 请替换为您的 DeepSeek API 密钥
            self.base_url = ""  # VOLCANO API 端点
            self.client  = Ark(api_key=os.getenv("volcano_api_key"))

        elif self.provider == LLMSource.AWS:
            self.aws_region = "us-west-2"  # 请替换为您的 AWS 区域（如 "us-east-1"）
            self.client = boto3.client("bedrock-runtime", region_name=self.aws_region, 
                                       aws_access_key_id=os.getenv("aws_access_key_id"), aws_secret_access_key=os.getenv("aws_secret_access_key"))

        # 📌 Hugging Face (`transformers`)
        elif self.provider == LLMSource.HUGGINGFACE:
            print(f"🚀 加载 Hugging Face 模型: {self.model.model_name}")
            path_huggingface_cache = os.getenv("huggingface_cache_location")
            print(f"👌确认 Huggingface Cache 位置:{path_huggingface_cache}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name,  cache_dir=path_huggingface_cache, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model.model_name,  cache_dir=path_huggingface_cache, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
            print("\n💬 模型加载完成。")     
        else:
            raise ValueError("❌ 无效的供应商，请使用 Model 枚举选择模型！")

    def chat(self, prompt, stream=True):
        """
        处理 AI 交互，支持流式模式和非流式模式
        :param prompt: 用户输入的提示词
        :param stream: 是否使用流式响应（True = 逐词输出, False = 一次性返回完整答案）
        """
        if stream:
            self._chat_stream_word_by_word(prompt)
        else:
            return self._chat_non_stream(prompt)

    def is_chinese(self, text):
        """检查文本是否包含中文字符"""
        return re.search(r'[\u4e00-\u9fff]', text) is not None
    

    def stream_response(self, response):
        """
        逐词流式打印响应
        - 中文：不加空格
        - 英文：单词之间加空格
        """
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content  # 获取文本
                
                if self.is_chinese(text):
                    print(text, end="", flush=True)  # 中文字符直接拼接
                else:
                    words = text.split()  # 按空格拆分单词
                    for word in words:
                        print(word, end=" ", flush=True)  # 英文单词之间加空格
                
                time.sleep(self.streaming_displace_delay_time)  # 轻微延迟，模拟 AI 打字效果
        print()  # 结束换行

    def _chat_stream_word_by_word(self, prompt):
        """
        📌 逐词流式响应（适用于 OpenAI / DeepSeek / AWS Bedrock）
        :param prompt: 用户输入的提示词
        """
        if self.provider in [LLMSource.OPENAI, LLMSource.DEEPSEEK, LLMSource.VOLCANO, LLMSource.ZHIPUAI, LLMSource.QWEN]:
            response = self.client.chat.completions.create(
                model=self.model.model_name,
                messages=[{"role": "system", "content": "你是一个智能助手。"},
                        {"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=True
            )
            print("\n💬 AI 逐词响应：")
            self.stream_response(response)

        if self.provider in [LLMSource.HUGGINGFACE]:
            """
            📌 Hugging Face (`transformers`) 生成文本
            """
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_length=150, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print("\n💬 AI 逐词响应：")
            for word in response:
                print(word, end=" ", flush=True)
                time.sleep(self.streaming_displace_delay_time)
            print()

        elif self.provider == LLMSource.AWS:
            if "claude" in self.model.model_name:

                body = {
                    "anthropic_version": "bedrock-2023-05-31",  # ✅ Claude 3.5 需要指定版本
                    "messages": [{"role": "user", "content": prompt}],  # ✅ Claude 3.5 需要 `messages`
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9
                }

                response = self.client.invoke_model(
                    modelId= self.model.model_name,  # ✅ Claude 3.5 Haiku
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body)
                )

                response_body = json.loads(response["body"].read().decode("utf-8"))
                
                # ✅ Claude 3.5 返回的数据格式
                result =  response_body["content"][0]["text"]

                # body = json.dumps({
                #     "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                #     "max_tokens_to_sample": 1024,
                #     "temperature": self.temperature
                # })


                # body = {
                #     "anthropic_version": "bedrock-2023-05-31",  # ✅ Claude 3.5 需要指定版本
                #     "messages": [{"role": "user", "content": prompt}],  # ✅ Claude 3.5 需要 `messages`
                #     "max_tokens": 1024,
                #     "temperature": self.temperature,
                #     "top_p": 0.9
                # }

                # response = self.client.invoke_model(
                #     modelId= self.model.model_name,  # ✅ Claude 3.5 Haiku
                #     contentType="application/json",
                #     accept="application/json",
                #     body=json.dumps(body)
                # )

                # response_body = json.loads(response["body"].read().decode("utf-8"))
                
                # # ✅ Claude 3.5 返回的数据格式
                # result =  response_body["content"][0]["text"]

            elif "llama" in self.model.model_name:
                body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 1024,
                    "temperature": self.temperature,
                    "top_p": 0.9
                })

                #
                response = self.client.invoke_model(
                    modelId=self.model.model_name,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )
                response_body = response["body"].read().decode("utf-8")
                response_json = json.loads(response_body)

                # 🔍 **打印 AWS API 返回的完整 JSON**
                # print("🔍 AWS API 返回的完整 JSON：", json.dumps(response_json, indent=2, ensure_ascii=False))

                # ✅ **自动检测正确的 AI 响应字段**
                possible_keys = ["completion", "output", "generation", "text"]  # 兼容不同模型
                result = None
                for key in possible_keys:
                    if key in response_json:
                        result = response_json[key]
                        break  # 找到匹配字段就停止循环

                if result is None:
                    raise ValueError("❌ AI 响应格式不匹配！请检查 AWS 返回的 JSON 结构。")
                #

            else:
                raise ValueError(f"❌ 不支持的 AWS Bedrock 模型: {self.model.model_name}")



            words = result.split()
            print("\n💬 AI 逐词响应：")
            
            for word in words:
                print(word, end=" ", flush=True)
                time.sleep(self.streaming_displace_delay_time)
            print()



    def _chat_non_stream(self, prompt):
        """
        📌 一次性返回完整 AI 响应（非流式模式）
        :param prompt: 用户输入的提示词
        :return: AI 生成的完整文本
        """
        if self.provider in [LLMSource.OPENAI, LLMSource.DEEPSEEK, LLMSource.VOLCANO, LLMSource.ZHIPUAI, LLMSource.QWEN]:
            response = self.client.chat.completions.create(
                model=self.model.model_name,
                messages=[{"role": "system", "content": "你是一个智能助手。"},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False  # 关闭流式模式
            )
            return response.choices[0].message.content
     
                
        if self.provider in [LLMSource.HUGGINGFACE]:
            """
            📌 Hugging Face (`transformers`) 生成文本
            """
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_length=150, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response


        elif self.provider == LLMSource.AWS:


            if "claude" in self.model.model_name:

                body = {
                    "anthropic_version": "bedrock-2023-05-31",  # ✅ Claude 3.5 需要指定版本
                    "messages": [{"role": "user", "content": prompt}],  # ✅ Claude 3.5 需要 `messages`
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9
                }

                response = self.client.invoke_model(
                    modelId= self.model.model_name,  # ✅ Claude 3.5 Haiku
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body)
                )

                response_body = json.loads(response["body"].read().decode("utf-8"))
                
                # ✅ Claude 3.5 返回的数据格式
                result =  response_body["content"][0]["text"]


            elif "llama" in self.model.model_name:
                body = json.dumps({
                    "prompt": prompt,
                    "max_gen_len": 1024,
                    "temperature": self.temperature,
                    "top_p": 0.9
                })
                #
                response = self.client.invoke_model(
                    modelId=self.model.model_name,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )

                response_body = response["body"].read().decode("utf-8")
                response_json = json.loads(response_body)

                # 🔍 **打印 AWS API 返回的完整 JSON**
                # print("🔍 AWS API 返回的完整 JSON：", json.dumps(response_json, indent=2, ensure_ascii=False))

                # ✅ **自动检测正确的 AI 响应字段**
                possible_keys = ["completion", "output", "generation", "text"]  # 兼容不同模型
                result = None
                for key in possible_keys:
                    if key in response_json:
                        result = response_json[key]
                        break  # 找到匹配字段就停止循环

                if result is None:
                    raise ValueError("❌ AI 响应格式不匹配！请检查 AWS 返回的 JSON 结构。")
                #

            else:
                raise ValueError(f"❌ 不支持的 AWS Bedrock 模型: {self.model.model_name}")

            return result
        
class Helper:

    def __init__(self):
        pass

    def run_ai_tests(self, user_input, models_to_test, k=1, output_file=""):
        """
        运行多个 AI 模型，并重复 K 次获取响应，存储到 CSV 文件
        :param user_input: 用户输入的提示词
        :param models_to_test: 需要测试的模型列表
        :param k: 每个模型重复执行的次数
        :param output_file: 保存 CSV 的文件名
        :return: 结果 Pandas DataFrame
        """
        # 📌 存储所有结果
        results = []

        # 📌 存储已初始化的模型（避免重复初始化）
        model_instances = {}

        # 📌 遍历多个模型
        for model in models_to_test:
            print(f"\n🔍 运行 {model.model_name} {k} 次...")

            # 🔹 仅在模型未初始化时创建 AI 客户端
            if model not in model_instances:
                model_instances[model] = AIChatClient(model=model)

            # 🔹 获取已初始化的模型
            ai_client = model_instances[model]

            # 🔹 运行 K 次
            for i in range(1, k + 1):
                print(f"🔄 第 {i} 次调用 {model.model_name} ...")

                # 🔹 获取 AI 生成的完整响应
                try:
                    full_response = ai_client.chat(user_input, stream=False)
                    print(f"📝 {model.model_name} 第 {i} 次返回: {full_response[:50]}...")  # 仅显示前50个字符
                except Exception as e:
                    full_response = f"❌ 发生错误: {str(e)}"
                    print(full_response)

                # 📌 记录结果
                results.append({
                    "Iteration": i,  # 第几次调用
                    "Model": model.model_name,
                    "Provider": model.provider.value,
                    "Response": full_response
                })

        # 📌 生成 Pandas DataFrame
        df = pd.DataFrame(results)

        # 📌 保存 DataFrame 到 CSV
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n✅ 所有模型响应已保存至 {output_file}！")

        return df
