import joblib
import numpy as np
model2 = joblib.load("model3.pkl")
x = np.array([[106/366,81/106,119/173,16/360]]) # Actual 128         85       142       45
print(x.shape)
val = model2.predict(x)
val = np.ceil(val*360)
val = val.astype(int)
print(val)
# input 106,81,119,16
# 109  90 130  42 model3
# 40 84 79 22 model2
# 40 84 79 22 model
# 77  94 104  38 modelWithAugmentation
# 72  95 104  39 modelWithAugmentation2
# 79  97 100  40 modelWithAugmentation3