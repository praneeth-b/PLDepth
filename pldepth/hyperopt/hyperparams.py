
from hyperopt import hp


lr_dict = {
"lr":    hp.loguniform('lr',-8, 0),
#"multi": hp.uniform("mlp",0,0.5)
}

