import json
import time
import os
from flask import Flask, render_template, request, jsonify
from models.model_utils import (
    load_pretrained_models, predict_digit, init_models
)

# 初始化Flask应用
app = Flask(__name__)

# 加载预训练模型（启动时加载一次）
print('Loading pretrained models...')
mlp_model, cnn_model = load_pretrained_models()

# 定义模型信息
MODEL_INFO = {
    'mlp': {
        'name': '全连接网络(MLP)',
        'layers': 3,
        'params': 10.2,
        'accuracy': 97.2
    },
    'cnn': {
        'name': '卷积神经网络(CNN)',
        'layers': 6,
        'params': 1.2,
        'accuracy': 99.1
    }
}

# 首页路由（视图）
@app.route('/')
def index():
    """渲染画板页面"""
    return render_template('index.html')

# 识别接口（控制器）
@app.route('/recognize', methods=['POST'])
def recognize():
    """处理手写数字识别请求"""
    try:
        # 获取请求数据
        data = request.get_json()
        image_data = data.get('image')
        model_type = data.get('model_type', 'cnn')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # 选择模型并预测
        if model_type == 'mlp':
            result = predict_digit(mlp_model, image_data)
        else:
            result = predict_digit(cnn_model, image_data)

        # 构造响应
        response = {
            'prediction': result['prediction'],
            'time': result['infer_time'],
            'model_info': MODEL_INFO[model_type]['name'],
            'success': True
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 主函数
if __name__ == '__main__':
    # 确保static文件夹存在
    if not os.path.exists('./static'):
        os.makedirs('./static')
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)