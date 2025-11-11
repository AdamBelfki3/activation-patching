import torch

def get_next_prediction(prompt, model, show=False):
    with torch.no_grad():
        with model.trace(prompt, remote=not model.dispatched):
            logits = model.lm_head.output[0][-1]

            prediction_id = logits.argmax(dim=-1).save()
            prediction_str = model.tokenizer.decode(prediction_id).save()
            probability = torch.nn.functional.softmax(logits, dim=0)[prediction_id].item().save()

    if show:
        print(f"Next Prediction: {prediction_str} | Probability: {probability:.2f}")

    return prediction_id