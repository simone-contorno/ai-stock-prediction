def clear_zero_values(data, features):
    # Clear zero values
    print((data==0).sum())
    for feat in data.columns:
        data.drop(data[data[feat] == 0].index, inplace=True)
    print("Data shape:", data.shape)
    print((data==0).sum())
    
    data = data[features]
    print(data.head())
    print(data.tail())

    return data