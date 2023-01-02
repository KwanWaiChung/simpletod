###  generate_dialogue_act_only_multiwoz.py
This slightly modifies from `generate_dialogue.py`. It only generates dialogue acts with ground truth belief state but without db states.

### generate_dialogue_act_only_sgd.py
Modifies from `generate_dialogue_act_only_multiwoz.py` which generates in sgd instead.

### generate_dialogue_act_only_sgd_batch.py
Modifies from `generate_dialogue_act_only_sgd.py` which generates in batch. The previous one can only generate one instance at a time and thus quite slow.