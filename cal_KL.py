import numpy as np
import 

# 计算两个概率list的相对熵
def calculateKL(prob_list_1, prob_list_2):
    KL = 0.0
    for i, prob_a in enumerate(prob_list_1):
        KL +=  prob_a * np.log(prob_a / prob_list_2[i], 2)

    return KL

if __name__ == "__main__":
    print("[DEBUG] Test Begin")
    prob_list_1 = [0.2, 0.3, 0.3, 0.2]
    prob_list_2 = [0.01, 0.49, 0.01, 0.49]
    print("KL Result: ", calculateKL(prob_list_1, prob_list_2))
    