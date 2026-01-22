$(document).ready(function() {
    // 画板初始化
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // 设置画笔样式
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000';
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 绘制事件
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });

    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mouseout', () => isDrawing = false);

    // 触摸适配（可选）
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        [lastX, lastY] = [touch.clientX - rect.left, touch.clientY - rect.top];
        isDrawing = true;
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        draw({
            offsetX: touch.clientX - rect.left,
            offsetY: touch.clientY - rect.top
        });
    });

    canvas.addEventListener('touchend', () => isDrawing = false);

    // 绘制函数
    function draw(e) {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    // 清空画板
    $('#clearBtn').click(() => {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        $('#resultBox').text('请书写数字并点击识别').css('font-size', '1.2rem');
        $('#modelInfo').text('模型信息：暂无');
    });

    // 识别数字
    $('#recognizeBtn').click(async () => {
        // 获取画布数据并转换为28x28灰度图
        const imgData = get28x28Grayscale(canvas);
        const modelType = $('#modelSelect').val();

        // 显示加载状态
        $('#resultBox').text('识别中...');
        $('#modelInfo').text(`模型信息：${modelType === 'cnn' ? '卷积神经网络' : '全连接网络'}`);

        try {
            // 提交到后端识别
            const response = await $.ajax({
                url: '/recognize',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    image: imgData,
                    model_type: modelType
                })
            });

            // 展示结果
            $('#resultBox').text(response.prediction).css('font-size', '3rem');
            $('#modelInfo').text(`模型信息：${response.model_info} | 识别耗时：${response.time}ms`);
        } catch (error) {
            $('#resultBox').text('识别失败');
            $('#modelInfo').text(`错误：${error.responseJSON?.message || '未知错误'}`);
        }
    });

    // 画布转28x28灰度图
    function get28x28Grayscale(canvas) {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;

        // 缩放并转换为灰度
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        const grayscale = [];

        // RGBA转灰度（取G通道，反转颜色：白色背景黑色笔迹→黑色背景白色笔迹）
        for (let i = 0; i < data.length; i += 4) {
            const gray = 255 - data[i + 1]; // 反转颜色
            grayscale.push(gray / 255.0); // 归一化到0-1
        }

        return grayscale;
    }

    // 初始化模型对比可视化图表
    initCharts();

    // 初始化图表函数
    function initCharts() {
        // 准确率图表
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        new Chart(accuracyCtx, {
            type: 'bar',
            data: {
                labels: ['全连接网络(MLP)', '卷积神经网络(CNN)'],
                datasets: [{
                    label: '测试集准确率(%)',
                    data: [97.2, 99.1],
                    backgroundColor: ['#e74c3c', '#2ecc71'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: '模型准确率对比' },
                    legend: { display: false }
                },
                scales: { y: { beginAtZero: true, max: 100 } }
            }
        });

        // 识别时间图表
        const timeCtx = document.getElementById('timeChart').getContext('2d');
        new Chart(timeCtx, {
            type: 'bar',
            data: {
                labels: ['全连接网络(MLP)', '卷积神经网络(CNN)'],
                datasets: [{
                    label: '单张识别时间(ms)',
                    data: [1.5, 3.2],
                    backgroundColor: ['#f39c12', '#3498db'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: '模型识别耗时对比' },
                    legend: { display: false }
                },
                scales: { y: { beginAtZero: true } }
            }
        });

        // 参数量图表
        const paramsCtx = document.getElementById('paramsChart').getContext('2d');
        new Chart(paramsCtx, {
            type: 'bar',
            data: {
                labels: ['全连接网络(MLP)', '卷积神经网络(CNN)'],
                datasets: [{
                    label: '参数量(万)',
                    data: [10.2, 1.2],
                    backgroundColor: ['#9b59b6', '#1abc9c'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: '模型参数量对比' },
                    legend: { display: false }
                },
                scales: { y: { beginAtZero: true } }
            }
        });
    }
});