import requests
import os
import json


SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
if SILICONFLOW_API_KEY is None:
    print("SILICONFLOW_API_KEY 环境变量未设置")
else:
    print("SILICONFLOW_API_KEY:",1)

    
url = "https://api.siliconflow.cn/v1/messages"

payload = {
    "model": "deepseek-ai/DeepSeek-V3.1",
    "messages": [
        {
            "role": "user",
            "content": "你是谁"
        }
    ],
    "max_tokens": 8192,
    "system": "你是一个股票经纪人",
    "stop_sequences": [],
    "stream": False,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
}
headers = {
    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text,type(response.text))

response_dict = json.loads(response.text)
print(response_dict['content'][0]['text'])


