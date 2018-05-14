---
layout: post
title: "Precision-Recall curve VS Receiver-Operator-Characteristic curve"
author: "Kwangjin Yoon"
categories: 
  - machine__learning
  - performance__analysis
tags: 
  - Precision-Recall curve
  - Receiver-Operator-Characteristic curve
---

<b>참고</b>: Davis, J., & Goadrich, M. (2006, June). [The relationship between Precision-Recall and ROC curves.](https://dl.acm.org/citation.cfm?id=1143874) 
In Proceedings of the 23rd international conference on Machine learning (pp. 233-240). ACM.


-----
## 0. Prerequisite

우선 아래 표부터 보자. 참고한 논문에서 가져온 것이다.

<div style="text-align:center" markdown="1">
![Imgur](https://i.imgur.com/P59g3jf.png)
</div>

이 표는 confusion matrix라고 불리는데, 이진 분류기<sup>binary classifier</sup>의 성능을 나타낸다.
지금부터는 이진 분류기를 BC (binary classifier) 또는 시스템, 아니면 분류기라고 혼용하여 쓰겠다.
$$TP$$는 true positives로 데이터셋의 positive 샘플들 중 BC가 맞춘 개수다. $$FP$$는 false positives로 데이터셋의 negative 샘플 중 BC가 posivite로 잘못 분류한 개수다.
$$FN$$은 false negatives로 데이터셋의 positive 샘플들 중 시스템이 negative로 잘못 분류한 개수다. $$TN$$은 데이터셋의 negative 샘플 중 시스템이 제대로 분류한 개수다. 
따라서, confusion 행렬에서 $$FP$$와 $$FN$$가 분류기의 오류에 대한 지표를 보여준다.


그러면 위 행렬에서 행은 (rows) 시스템의 응답에 (response) 대한 정보를 담고 있고, 컬럼은 (columns) 실제 데이터셋에서 데이터의 분포에 대한 정보를 담고있다. 
즉, $$TP + FP$$는 BC가 true라고 응답한 총 개수가 된다. BC가 negative라고 응답한 개수는 $$FN+TN$$ 이다.
반면, $$TP+FN$$은 데이터셋에서 positive 샘플의 개수다. $$FP+TN$$은 데이터셋에서 negative 샘플의 개수다.
이로부터 Recall, Precision, True Positive Rate (TPR), False Positive Rate (FPR)는 다음과 같이 계산된다.

$$
\begin{eqnarray}
    Recall &=& \frac{TP}{TP+FN} \\
    Precision &=& \frac{TP}{TP+FP} \\
    TPR &=& \frac{TP}{TP+FN} \\
    FPR &=& \frac{FP}{FP+TN}
\end{eqnarray}
$$

우선 $$Recall \triangleq TPR$$ 임을 확인하자. Recall (TPR) 은 데이터셋의 positive 샘플들 ($$TP+FN$$) 중에서 시스템이 맞춘 ($$TP$$) 비율을 의미한다.
반면, Precision은 시스템의 positive 응답들 ($$TP+FP$$) 중 맞은 ($$TP$$) 비율이다. FPR은 데이터셋의 negative 샘플들 ($$FP+TN$$) 에서 오검출 ($$FP$$) 을 발생한 비율이다.

Receiver-Operator-Characteristic (ROC) 공간은 $$x$$축이 $$FPR$$ 이고 $$y$$축이 $$TPR$$인 공간이다. 
반면에, Preicision-Recall (PR) 공간은 $$x$$축이 $$Recall$$이고 $$y$$축이 $$Precision$$으로 이루어져있다.
앞서 얘기한 confusion matrix는 이 공간들 위의 점이 된다. 즉, 어떤 시스템의 출력 결과로 confusion matrix $$A$$가 만들어졌다고 하자. 
이로부터 $$FPR(A)$$를 계산할 수 있다. ($$FPR$$이 confusion matrix의 함수라고 보는 것)
마찬가지로 $$TPR(A)$$, $$Recall(A)$$, $$Precision(A)$$도 계산 된다.
그러면 점 $$(FPR(A), TPR(A))$$는 ROC공간위의 한 점이 되고, 점 $$(Recall(A), Precision(A))$$는 PR 공간의 한 점이 된다. 아래 ROC와 PR 커브 예시가 있다.

<div style="text-align:center" markdown="1">
![Imgur](https://i.imgur.com/Ja8xMW5.png)
</div>

이진 분류 문제에서 분류기의 출력은 일반적으로 연속적인 값을 가지며 여기에 임계치<sup>threshold</sup>를 적용해 클래스를 구별한다. 예를들어 시스템이 $$(-1, 1)$$ 사이의 값을 출력한다고 할때 임계치 $$\tau$$ 보다 높으면 positive 낮으면 negative로 분류하는 식이다.
그러면 어떤 임계치 $$\tau$$는 confusion matrix $$A_{\tau}$$를 만들게되고 이는 PR 이나 ROC 공간의 한 점이 된다.
임계치를 바꿔가며 PR (ROC) 공간에 점을 찍어서 PR (ROC) 커브를 그리면 된다.

## 1. Intro

어떤 객체의 클래스를 맞추는 (해당 클래스일 확률을 계산하는) 새로운 알고리즘을 개발 했다고 치자. (클래스 개수는 positive와 negative로 두 개라고하자, binary decision problem)
그런 알고리즘의 성능 평가를 위해서 단순히 정확도 (Accuracy)만 계산하며 평가를 하는것은 문제가 있다. ([참고](https://dl.acm.org/citation.cfm?id=657469)) 
참고의 말을 빌리자면 ROC 커브를 사용해야 정확한 성능 평가를 할 수 있다고 한다.
ROC 커브는 시스템이 정확히 맞추는 positive 샘플 개수가 어떻게 변화하는지를 시스템이 틀리는 negative 샘플의 개수를 변경해가며 보여준다.
ROC 공간에서 가장 좋은 위치는 $$(0,1)$$이며 따라서 커브가 좌측 상단에 붙을 수록 시스템의 성능이 좋다고 말 할 수 있다.
ROC 공간의 우측 상단은 $$FPR$$이 1이고 $$TPR$$도 1인 상태로 시스템이 모든 데이터 샘플에 대해서 (positive, negative 상관 없이) positive 응답을 한 경우다. 

PR 커브는 ROC 커브의 대안으로써 언급된다. 왜냐면 ROC 커브는 데이터의 분포에 따라 시스템의 성능 분석이 달라질수 있기 때문이다.
예를들어 ROC 공간의 점 $$(0.15, 0.5)$$을 생각해보자. $$FPR$$이 15%고 $$TPR$$이 50%인 성능을 나타낸다.
negative와 positive 샘플의 개수 정확히 반반으로 각 100개씩이라면 시스템은 15개의 false positive를 생성했을때 50개의 true positive를 생성한 것이다.
그러나 negative 샘플이 1000개 positive 샘플이 100개라면, 그 점에서 시스템은 150개의 false positive와 함께 50개 true positive를 생성한 것이 된다.
true positive 대비 꽤 많은 false positive를 생성했음에도 불구하고 ROC 공간 상에 같은 점으로 표시되게 된다.

PR 커브는 데이터의 분포가 편중된 상황에서도 분류기의 성능 평가를 하는데 자주 쓰인다.
PR 공간상에서 최고 위치는 우측 상단이다. Recall이 100%일때 Precision도 100%인 상황이다. 따라서 커브가 오른쪽 상단에 가까울수록 좋은 성능을 나타낸다.

위 그림의 ROC와 PR 커브는 두개의 알고리즘을 학습시킨 결과로 그린 것이다. 학습 데이터는 negative 샘플의 개수가 positive보다 상당히 많았다.
ROC 커브를 보면 알고리즘 1과 2 모두 꽤 잘 분류를 하고 있는 것으로 보인다. 그러나 PR 커브를 보면 둘다 아직 개선해야할 여지가 있어 보이고, ROC 커브에서는 보이지 않았던 알고리즘 1과 2의 성능 차이도 보인다.

## 2. ROC space와 PR space 사이의 관계

참고한 [논문](https://dl.acm.org/citation.cfm?id=1143874)에서 소개한 ROC 공간과 PR 공간의 연관성 세 가지를 정리한다.

1. $$Recall$$이 0이 아닌 경우라면, ROC 공간의 커브와 PR 공간의 커브는 일대일 대응이 존재한다.
2. ROC 공간에서 어떤 커브가 다른 커브보다 위에 있다면 (domiate) PR 공간에서도 위에 있다. 그리고 그 반대도 마찬가지다.
3. ROC 커브의 convex hull을 이용해 achievable PR curve를 그릴 수 있다.

위 세 가지에 대한 증명이 궁금하다면 직접 논문을 읽어보시라. 여기서는 각각이 무슨 의미가 있는지만 짚어보자.
우선 1은 ROC 커브를 PR 커브로 변환 할 수 있다는 얘기다. PR에서 ROC로도 가능하다. 리콜이 0인 경우는 매우 특이한 케이스를 얘기하는 것이니 크게 중요치 않다.
2에서 위에 있다 (dominate) 라는 얘기는 말그대로 모든 구간에서 어떤 커브가 다른 커브를 감싸고 있거나 최소 겹쳐있다는 얘기다.
그러면 2는 ROC 공간에서 위에 있는 커브는 PR 공간에서도 위에 있고, 반대로 PR 공간에서 위에 있는 커브는 ROC 공간에서도 위에 있다고 얘기하고 있다.
커브가 위에 있고 밑에 있고를 따지는 이유는 위에 있는 커브는 성능 좋은 분류기를 의미하기 때문이다.
3을 얘기하기위해 ROC 공간에서 convex hull이 무엇을 의미하는지 보자.

<div style="text-align:center" markdown="1">
![Imgur](https://i.imgur.com/80iuVpH.png)
</div>

위 그림 (a)를 보면 ROC 공간에 4개의 점이 있다. 어떤 알고리즘의 성능평가 결과로 4개의 confusion matrix를 얻은 것이고 그것으로 ROC 공간에 점을 찍은 것이다.
이 네개의 점을 (b) 처럼 선형 보간으로 이으면 ROC 커브를 얻게 된다. 근데 (a) 처럼 convex hull 그리면 알고리즘의 가능한 최대 성능을 알수 있다. 
왜냐하면 convex hull은 주어진 점을 이용해 그릴 수있는 모든 커브들 보다 항상 위에 (dominate) 있기 때문이다.
그러면 ROC 공간의 convex hull을 PR 공간으로 변환하면 (1번) PR 공간에서도 주어진 점으로 만들 수 있는 어떤 커브들 보다도 항상 위에 존재하는 (2번) achievable PR curve를 얻게 된다 (그림 (c)).

## 3. 결론

사실 이 글을 통해 하고자 했던 말은 이거 였다.
데이터가 편중되어 있다면 (highly skewed dataset) ROC 커브 말고 PR 커브를 통해 성능 평가를 해야 한다는 것이다.
참고한 논문에서도 등장했던 얘긴데, 데이터셋 클래스의 분포가 편중되어 있다면 ROC 커브는 분류기를 과대평가 하기 때문에 좋지 못하다는 것이다.
예를 들어 negative sample이 positive sample들 보다 매우 많은 경우를 생각해보자. 그런 데이터셋을 통해 분류기를 학습시키면 negative sample을 많이 보았기 때문에 negative 반응률이 좋을 것이다.
또한 negative sample의 수가 많기 때문에 어지간히 많은 $$FP$$를 발생하지 않는 이상 $$FPR$$이 높아지지 않아 커브가 왼쪽으로 치우쳐지게 된다. 결과적으로 분류기 성능이 좋게 보이는 것이다.

끝으로 논문에서 언급한 ROC 공간과 PR 공간의 연관성 세 가지도 연구할때 참고하여 잘 이용하면 좋을 것이다. 그리고 논문에서는 PR 공간에서 interpolation 하는 법을 소개하고있다.
또 ROC 공간에서 AUC (Area Under Curve)를 최적화한다고 해서 PR 공간에서도 AUC가 최적화 되는 것은 아니라고 논문에서 보여준다.
필요한 사람은 논문을 참고하면 되겠다.
