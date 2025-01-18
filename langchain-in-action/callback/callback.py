def compute(x, y,callback):
    result = x * y
    callback(result)



def print_result(result):
    print(f"The result is: {result}")

def square_result(result):
    print(f"The square result is: {result * result}")
# 使用print_result作为回调
compute(1, 2, print_result)

# 使用square_result作为回调
compute(2, 2, square_result)
