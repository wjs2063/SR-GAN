
# SR-GAN
 (실행환경) GOOGLE COLAB  
 (수행기간) 2021-07-05~2021-07-20


## 기본적인 GAN 의 Modeling 




<img width="500" alt="스크린샷 2021-07-30 오전 1 01 06" src="https://user-images.githubusercontent.com/76778082/127525921-21cea387-8675-4241-81c0-03adc511f24c.png">


1. generator 가 이미지를 생성: Fake image -> 진짜 이미지와 Fake image  
2. dataset에있는 진짜이미지 Real image                                             -------> discriminator 가 판별한다 그리고 역전파를 시킨다 


Generator 의 목적은 discriminator 를 속여 진짜같은 이미지를 만들어내는것 
Discriminator 의 목적은 가짜이미지를 검출해내는것

이와같이 모델링이 짜여져있다.

이 모델의 목적은  서로의 확률이 0.5 0.5 인 내쉬균형을 이루도록 하는것이다.

하지만 이를 달성하기에는 몇가지 어려움이있다

3가지정도의 문제점을 살펴보자 


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



### Load Data set
```
def load_dataset(train_all_input_files:list,train_all_label_files:list,a:int,
                 b:int)->list:
    final_train_input=[]
    final_label_input=[]
    for index in range(a,b):
        ## train 이미지를 list에 담는다
        train_path=train_all_input_files[index]
        
        train_input=image.load_img(train_path,target_size=(256,256))
        train_input=image.img_to_array(train_input)
        train_input=np.array(train_input,dtype=np.float32)

        label_path=train_all_label_files[index]
        label_input=image.load_img(label_path,target_size=(256,256))
        label_input=image.img_to_array(label_input)
        label_input=np.array(label_input,dtype=np.float32)
       
        ## label 이미지를 list에 담는다
        final_train_input.append(train_input)
        final_label_input.append(label_input)
        
        #train_input 저해상도 label_input 고해상도
    return final_train_input,final_label_input

def load_testset(test_all_input_files:list)->list:
    test_data=[]
    for index in range(len(test_all_input_files)):
        test_path=test_all_input_files[index]
        test_input=image.load_img(test_path,target_size=(256,256))
        test_input=image.img_to_array(test_input)
        test_input=np.array(test_input,dtype=np.float32)
        
        
        #test_input=test_input/127.5-1 #normalization을 해준다
        test_data.append(test_input)
    return test_data
```


### Build Model

```
class Generator():
    def __init__(self,input_shape):
        self.input_shape=(256,256,3)

    def residual_block(self,x):
    
        filters = (64, 64)
        kernel_size = 3
        strides = 1
        padding = "same"
        momentum = 0.8
        activation = "relu"

        resid = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
        resid = Activation(activation=activation)(resid)
        resid = BatchNormalization(momentum=momentum)(resid)

        resid = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(resid)
        resid = BatchNormalization(momentum=momentum)(resid)

        
        resid = Add()([resid, x])
        return resid

    def build_generator(self):
        ## 함수형 API를 사용하여 작성하였다
        n_residual_blocks = 16
        momentum = 0.8
        

        # input 층 작성
        input_layer = Input(shape=self.input_shape)

        # 첫시작부분
        gen1 = Conv2D(filters=64, kernel_size=9,
                      strides=2, padding='same')(input_layer)
        
        # residual block 을 15개층을 쌓아줍니다.
        resid = self.residual_block(gen1)
        for i in range(n_residual_blocks - 1):
            resid = self.residual_block(resid)

        # CNN 구조와 배치정규화를 시켜줍니다. 이유는 GAN 의 불안정한 학습을 막기위해.
        gen2 = Conv2D(filters=64, kernel_size=3, strides=1, 
                      padding='same')(resid)
        gen2 = BatchNormalization(momentum=momentum)(gen2)

        
        gen3 = Add()([gen2, gen1])

        # Activation 함수는  PreLU 로도 써도될것같다.
        gen4 = UpSampling2D(size=2)(gen3)
        gen4 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(gen4)
        gen4 = LeakyReLU(0.1)(gen4)

        gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
        gen4 = LeakyReLU(0.1)(gen4)

        gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
        gen4 = LeakyReLU(0.1)(gen4)

        gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
        gen4 = LeakyReLU(0.1)(gen4)
        gen5 = UpSampling2D(size=2)(gen4)
        gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
        gen5 = Activation('relu')(gen5)

        
        gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
        output = Activation('tanh')(gen6)
        model = Model(inputs=[input_layer], outputs=[output], name='generator_model')
        return model



def build_discriminator():
    # discriminator 또한 함수형 API 를 이용해 만들었다
    leakyrelu_alpha = 0.2
    momentum = 0.8
    
    input_shape=(None,None,3)
    input_layer = Input(shape=input_shape)

    
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    
    dis2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    
    dis4 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    
    dis6 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    
    dis8 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)

    # 마지막이 Dense 1로 넣어준이유는 discriminator 는 분류만한다.
    output = Dense(units=1, activation='sigmoid')(dis9)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    
    return model
    
```
### Pre trained RESNET 불러오기

```
def build_resnet():
    
    input_shape = (None, None, 3)

    # 사전학습된 ResNet 으로불러오자
    res =tf.keras.applications.ResNet152V2(weights="imagenet",include_top=False, input_shape=(256, 256, 3))
    
    
    
    
    
    
    
    ## 모델생성~
    model = Model(inputs=res.inputs, outputs=res.layers[9].output)
    model.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])
    
    return model
```



### Adversirial 구축 및 Training 


```
epochs = 30000
batch_size = 4


## image shape 지정한다
low_resolution_shape = (256, 256, 3)
high_resolution_shape = (256, 256, 3)

## resnet 을 훈련중지시킨다. 이유는 더이상 학습하면 안되고 특징추출의 기능만 하게하기위함이다
common_optimizer = Adam(0.0002, 0.5)
res = build_resnet()
res.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])
res.summary()
res.trainable = False



# dicriminator 생성
shape=(256,256,3)
#discriminator =load_model("/content/drive/MyDrive/LG_result/720srdiscriminator5000.h5")
discriminator=build_discriminator()
discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

discriminator.summary()


##generator 생성
##generator = load_model("/content/drive/MyDrive/LG_result/720srgenerator5000.h5")
c=Generator(shape)
generator=c.build_generator()
generator.summary()



## adversarial 모델을 만들어보자


## 저해상도,고해상도 Input layers 를 만듬
input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)

## 저해상도에서 고해상도이미지 생성
generated_high_resolution_images = generator(input_low_resolution)

## generator 가생성해낸 이미지의 특징추출 
features = res(generated_high_resolution_images)

## discriminator 훈련 중지
discriminator.trainable = False

## probability 저장
probs = discriminator(generated_high_resolution_images)

## adversarila 모델 생성
adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer,)




for epoch in range(epochs):
    print("Epoch:{}".format(epoch))

    """
    Train the discriminator network
    """

    
    high_resolution_images, low_resolution_images = sample_images(batch_size=batch_size,
                                                                  low_resolution_shape=low_resolution_shape,
                                                              high_resolution_shape=high_resolution_shape)
    
    # 데이터 전처리
    high_resolution_images = high_resolution_images / 127.5 - 1.

    low_resolution_images = low_resolution_images / 127.5 - 1.

    # low resolution image 로 high resolution 을 만들어냄
    generated_high_resolution_images = generator.predict(low_resolution_images)

    # real 과 label 을 만들 크기생성 
    real_labels = np.ones((batch_size, 256, 256, 1))
    fake_labels = np.zeros((batch_size, 256, 256, 1))
    
    # discriminator 학습
    d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)

    #  discriminator 손실함수의 합
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    print("d_loss:", d_loss)

    ## network 훈련

    ## batch_size 만큼 샘플추출한다.
    high_resolution_images, low_resolution_images = sample_images( batch_size=batch_size,
                                                                  low_resolution_shape=low_resolution_shape,
                                                                  high_resolution_shape=high_resolution_shape)
    
    high_resolution_images = high_resolution_images / 127.5 - 1.
    low_resolution_images = low_resolution_images / 127.5 - 1.

    ## vgg 를 활용한 feature map 추출
    image_features = res.predict(high_resolution_images)
    

    ## generator network 훈련
    g_loss = adversarial_model.train_on_batch([low_resolution_images, high_resolution_images],
                                     [real_labels, image_features])

    print("g_loss:", g_loss)

    
    
    generated_images = generator.predict_on_batch(low_resolution_images)
           
    # epoch 마다 저장을 해놓아야함 아래수치마다 저장함
    if epoch % 100 == 0:
        
        # Normalize images
        high_resolution_images, low_resolution_images = sample_images(batch_size=batch_size,
                                                                low_resolution_shape=low_resolution_shape,
                                                                high_resolution_shape=high_resolution_shape)
                # Normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        generated_images = generator.predict_on_batch(low_resolution_images)

        for index, img in enumerate(generated_images):
            visualize_image(img)
        
        
    if epoch%1000==0 and epoch!=0:
        

        for index, img in enumerate(generated_images):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path="/content/drive/MyDrive/LG_result/img_{}_{}".format(epoch, index))
    if epoch%2500==0 and epoch!=0:
        generator.save_weights("/content/drive/MyDrive/LG_result/720srgenerator.h5")
        discriminator.save_weights("/content/drive/MyDrive/LG_result/720srdiscriminator.h5")
        generator.save(f"/content/drive/MyDrive/LG_result/720srgenerator{epoch}.h5")
        discriminator.save(f"/content/drive/MyDrive/LG_result/720srdiscriminator{epoch}.h5")


## 모델저장

generator.save_weights("/content/drive/MyDrive/LG_result/720srgenerator.h5")
discriminator.save_weights("/content/drive/MyDrive/LG_result/720srdiscriminator.h5")
generator.save(f"/content/drive/MyDrive/LG_result/720srgenerator{epochs}.h5")
discriminator.save(f"/content/drive/MyDrive/LG_result/720srdiscriminator{epochs}.h5")
```
















코드실행결과 
# epoch==0
<img width="363" alt="스크린샷 2021-07-29 오후 5 40 16" src="https://user-images.githubusercontent.com/76778082/127462958-363cd299-71df-4ddc-be52-1260f8ae9a37.png">

# epoch==100
<img width="363" alt="스크린샷 2021-07-29 오후 6 11 51" src="https://user-images.githubusercontent.com/76778082/127465341-88f58725-30b1-4ed7-beb5-e89c2dd5ebe0.png">


# epoch==200


<img width="363" alt="스크린샷 2021-07-29 오후 6 10 34" src="https://user-images.githubusercontent.com/76778082/127465194-f4b72e03-72c5-4d3e-a909-b15bf79cdd74.png">

# epoch==5500

<img width="363" alt="스크린샷 2021-07-30 오후 6 21 32" src="https://user-images.githubusercontent.com/76778082/127632114-53ee4bee-9fce-44ba-93bb-8e2f39061e1b.png">




## epoch 을 20000만번 이상  나름 괜찮은 quality 의 사진이나온다


## 어려웠던점:
0. 데이터 전처리 시 cv2.imread 와 load_img 의 불러오는 방식이 틀려서 애를먹었다. cv2 는 channel 을 BGR 로 불러오게되고 다시 RGB채널로 바꾸어야한다. load_img 는 처음부터 RGB로 불러온다
1. 모델이 크다보니까 shape 를 잘못맞추게되면 찾기가 굉장히 힘들어진다. shape 을 손으로 다 일일히 계산해서 찾거나 print 찍어야했다.
2. Generator,Discriminator 균형을 맞추는게 생각보다 힘들다. 처음에 Resnet 과 Unet 을 섞어만든 ResUnet 으로 generator 를 생성해보았는데 좋은결과가 나오지않았다.
3. Artifact 가 생긴다. discriminator 가 더빨리학습될떄 생긴다는 의견도있다. 이에 대한 해결방법으로는 stride 와 kernel size 가 나누어 떨어지게하는방법등이있는데 정확히는 모르겠다.
GAN 분야는 아직도 활발하게 연구가되고있는 분야여서 아직 정확한 해결방법이없다.





해당 전체코드는 jpynb 에있습니다. 궁금하신점은 질문 남겨주세요.
 
