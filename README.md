
# SR-GAN




## 기본적인 GAN 의 Modeling 




<img width="781" alt="스크린샷 2021-07-30 오전 1 01 06" src="https://user-images.githubusercontent.com/76778082/127525921-21cea387-8675-4241-81c0-03adc511f24c.png">


1. generator 가 이미지를 생성: Fake image -> 진짜 이미지와 Fake image  
2. dataset에있는 진짜이미지 Real image                                             -------> discriminator 가 판별한다  역전파를 시킨다 


Generator 의 목적은 discriminator 를 속여 진짜같은 이미지를 만들어내는것 
Discriminator 의 목적은 가짜이미지를 검출해내는것

이와같이 모델링이 짜여져있다.

자세히 살펴보면 결귝 가장 이상적인 분포는 서로의 확률이 0.5 0.5 인 내쉬균형을 이루는 점이다. 

하지만 이상적인경우고 이를 달성하기란 쉽지않다.

아래의 문제점이있다.

1. Mode Collapse : 최빈값 붕괴 
원인: GAN 을 훈련할떄 discriminator 에 독립적인 입력을 전달하게된다면 제대로 학습을 하지못하게된다 (경사도?문제같음)이러한 것이 model collapse 이다


2. Vanishing gradient: 경사소멸
원인:역전파가 되는 동안 경사도는 마지막계층에서 처음계층으로 흐른다. 경사도가 역방향으로 흐르는동안에 경사도는 줄어들게 되며 너무 작아지면 학습이 불가능함.
해결책: ReaKy ReLU ReLu PReLU 등의 활성함수를활용하면 해결가능함.

3. internal covariate shift : 내부 공변량  변화

입력 분포가 변경되면 새로운분포에 적응하는 법을 학습하려고한다 -> 훈련과정이느려짐-> global minimum에 수렴하는 시간이 줄어든다. 

해결책: BatchNormalization 또는 그밖의 정규화기법을 적용하면 해결가능하다고한다.


### GAN 의 안정성 해결:

Minibatch판별 ,역사적 평균,단측레이블 평활화


이정도로 간략히 요약한다.




코드실행결과 
# epoch==0
<img width="363" alt="스크린샷 2021-07-29 오후 5 40 16" src="https://user-images.githubusercontent.com/76778082/127462958-363cd299-71df-4ddc-be52-1260f8ae9a37.png">

# epoch==100
<img width="363" alt="스크린샷 2021-07-29 오후 6 11 51" src="https://user-images.githubusercontent.com/76778082/127465341-88f58725-30b1-4ed7-beb5-e89c2dd5ebe0.png">


# epoch==200


<img width="363" alt="스크린샷 2021-07-29 오후 6 10 34" src="https://user-images.githubusercontent.com/76778082/127465194-f4b72e03-72c5-4d3e-a909-b15bf79cdd74.png">

# epoch==5500

<img width="363" alt="스크린샷 2021-07-30 오후 6 21 32" src="https://user-images.githubusercontent.com/76778082/127632114-53ee4bee-9fce-44ba-93bb-8e2f39061e1b.png">




## 이와같이 epoch 을 20000만번 이상  나름 괜찮은 quality 의 사진이나온다


해당 전체코드는 jpynb 에있습니다. 궁금하신점은 질문 남겨주세요.
 
