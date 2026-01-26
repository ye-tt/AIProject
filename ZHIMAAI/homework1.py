import requests

def get_response_from_model(model, mycontent):
    url="https://api.siliconflow.cn/v1/chat/completions"
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-cumwktcvhnxyqbqnhqjsoiknoozfklowfcdglkqvikohtjsl"
    }
    payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": mycontent
        }
    ]
}
    response = requests.post(url, headers=headers, json=payload)
    print(f"status: {response.status_code}")
    return response.json()


conents ="请用一句总结以下内容：人工智能对未来发展具有重大意义，应该使之为各国各地区人民造福。我们要以全人类福祉为念，推动人工智能朝着有益、安全、公平方向健康有序发展。中国倡议成立世界人工智能合作组织，希望通过发展战略、治理规则、技术标准等合作，积极为国际社会提供人工智能公共产品。中方愿同亚太经合组织各成员一道，共同提升民众人工智能素养，弥合亚太地区数字和智能鸿沟。"
mymodel = "Qwen/QwQ-32B"

resResult = get_response_from_model(mymodel, conents)
print(f"result: + {resResult}")


#第二题
conents2 ="新的问题，请判断以下评论的情感倾向是正面、负面还是中立： '这家餐厅服务态度很差，但味道非常地道。"
mymodel2 = "deepseek-ai/DeepSeek-V3.2-Exp"
resResult2 = get_response_from_model(mymodel2, conents2)
print(f"new result: + {resResult2}")
