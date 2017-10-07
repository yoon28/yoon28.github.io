---
layout: post
title: "Back Propagation and Auto Differentiation"
author: "Kwnagjin Yoon"
tags: deep-learning back-propagation auto-differentiation
---


Back propagation(BP)은 딥러닝에서 뉴런들의 Weight를 업데이트하는데 쓰이는 알고리즘이고 Auto differentiation(AD)은 컴퓨터가 미분을 할수 있게 해주는 알고리즘 중 하나다. 텐서플로우, 파이토치에서 BP가 AD를 이용하여 구현되었다고 함.

참고:
[link1](https://en.wikipedia.org/wiki/Backpropagation)
[link2](http://neuralnetworksanddeeplearning.com/chap2.html)
[link3](https://www.suchin.co/2017/03/18/Automatic-Differentiation-and-Backpropagation/)

-----

## Back propagation

뉴럴넷 $$N$$ 이, $$e$$ 개의 연결을 가지고 있고, $$m$$ 개의 입력을 취하며, $$n$$ 개의 출력을 가진다고하자. 즉, 어떤 트레이닝 샘플 $$(x_i,y_i)$$가 $$x_i \in R^m, y_i \in R^n$$ 이고, $$i$$는 트레이닝 샘플의 인덱스다. 뉴럴넷 $$N$$의 가중치 $$w_j \in R^e$$ 는 back propagation 알고리즘을 통해 $$j=0,1,2,...$$ 순으로 업데이트 해가며 $$w_0$$ 는 초기 가중치 값이다. 

*작성 중...*