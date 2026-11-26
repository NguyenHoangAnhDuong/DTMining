import numpy as np
from config import H


def build_targets(data):
    """Horizon tính theo quan sát còn lại; không nội suy nhãn tương lai."""
    data = data.copy()
    price = data['Gold Price']
    for horizon in H:
        future_price = price.shift(-horizon)
        data[f'target_h{horizon}'] = future_price
        data[f'target_logret_h{horizon}'] = np.log(future_price / price)
        data[f'target_date_h{horizon}'] = data['Date'].shift(-horizon)
    return data
