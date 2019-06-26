## Predict_Diabetes_with_DNN
### 당뇨병을 예측하는 DNN 모델
공공데이터 포털에서 2016년 건강검진대상자 100만명에 대한 건겅검진정보를 이용하여 당뇨병을 예측
<br>
과적합이 일어나도록 은닉층의 개수는 4개 각층의 개수는 18개 20개 28개 9개로 잡음
<br>
과적합이 일어나도록 DNN 모델을 설정하고 배치정규화, 가중치 초기화, 드롭아웃을 적용시켜 그 결과값들을 비교함
<br>
하지만 과적합이 일어나지 않은것인지 정확도 자체는 높으나 배치정규화 가중치 초기화, 드롭아웃을 적용시키고 나서의 차이점은 딱히 없었음