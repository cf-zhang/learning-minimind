from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./minimind-3")
attention = model.model.layers[0].self_attn
for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    projection = getattr(attention, name)
    print(
        name,
        "weight shape =", projection.weight.shape,
        "参数量 =", projection.weight.numel(),
        "bias =", projection.bias
    )

