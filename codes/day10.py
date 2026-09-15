from transformers import AutoConfig

config = AutoConfig.from_pretrained("./minimind-3")

print("配置对象类型：", type(config).__name__)
print("model_type：", config.model_type)
print("architectures：", config.architectures)
print("hidden_size：", config.hidden_size)
print("num_hidden_layers：", config.num_hidden_layers)

