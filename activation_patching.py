import torch

def activation_patching( 
    source_prompt, 
    target_prompt,
    source_patch, 
    target_patch,
    model
):
    with torch.no_grad():

        with model.session(remote=not model.dispatched):

            with model.trace() as tracer:
                
                source_activations = list()
                with tracer.invoke(source_prompt)as invoker:
                    for i, layer in enumerate(model.model.layers):
                        if isinstance(layer.output, tuple):
                            source_activations.append(layer.output[0][0, source_patch, :].clone().detach())
                        else:
                            source_activations.append(layer.output[0, source_patch, :].clone().detach())

                    source_logits = model.lm_head.output[0][-1].detach().cpu().save()

            with model.trace() as tracer:

                with tracer.invoke(target_prompt):
                    target_logits_clean = model.lm_head.output[0][-1].detach().cpu().save()

                target_logits_patched = dict().save()
                for i in range(len(model.model.layers)):
                    with tracer.invoke(target_prompt):
                        if isinstance(model.model.layers[i].output, tuple):
                            model.model.layers[i].output[0][0, target_patch, :] = source_activations[i]
                        else:
                            model.model.layers[i].output[0, target_patch, :] = source_activations[i]

                        target_logits_patched[str(i)] = model.lm_head.output[0][-1].detach().cpu()

    return source_logits, target_logits_clean, target_logits_patched