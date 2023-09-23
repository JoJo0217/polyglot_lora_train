polyglot model의 8bit 양자화 lora training code입니다.   
tloen/alpaca-lora 코드를 바탕으로 kullm에서 수정한 코드의 peft weight 저장 오류, loss 반복 등을 고친 코드입니다.   
모델 결과는 tensorboard로 확인 가능하게 만들었기에 다음과 같이 확인이 가능합니다.   
```
tensorboard --logdir ./chpt/runs   
```   
모델 학습 후 testkullm.py로 gradio test 까지 가능하게 만들어 두었습니다.