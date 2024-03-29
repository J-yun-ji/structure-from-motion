<h1 align="center">SFM</h1>
<p align="center">
  <strong>Structure-from-Motion</strong>
<br>
2차원 영상으로부터 3차원 정보를 추출하여 3D로 재구성
<br>
structure = 3D structure, motion = Camera pose
</p>
<br>

### 🖇️ [실행결과 보러가기](https://github.com/J-yun-ji/structure-from-motion/blob/master/pdf/SIFT_%EC%A3%BC%ED%94%BC%ED%84%B0%20%EB%85%B8%ED%8A%B8%EB%B6%81%20%EC%8B%A4%ED%96%89%20%EA%B2%B0%EA%B3%BC.pdf)

## 🖇️ 목차
1. [SIFT vs ORB vs AKAZE](#비교)
2. [keypoint / descriptor](#키포인트) 
<br>
<br>



 

<h2 id="비교">01. SIFT vs ORB vs AKAZE</h2> 

<br>

### 📌 SIFT(Scale-Invariant Feature Transform)
**[장점]**
- 크기와 회전에 대해 불변성을 가지고 있어 다양한 변형에 강함.
- 정확한 매칭을 제공하며 높은 품질의 descriptor를 생성.
- 다양한 환경에서 안정적인 성능 제공.

**[단점]**
- 계산량이 많아 속도가 느림. 
- GPU 가속화를 지원하지 않음.
- 특징점 검출에 있어서 잡음에 민감할 수 있음

=> **정확한 매칭과 변형에 강한 특징이 필요한 경우**에 사용.

<br>

### 📌 ORB(Oriented FAST and Rotated BRIEF)
**[장점]**
- 빠른 속도를 가지고 있어 실시간 애플리케이션에 적합.
- 회전과 크기에 대한 불변성을 제공.
- descriptor를 사용하여 효율적인 매칭 제공.

**[단점]**
- 일부 변형에 대해서는 SIFT보다 불안정할 수 있음.
- 일반적으로 SIFT보다 매칭의 정확도가 낮을 수 있음.
- 대부분의 경우 상대적으로 작은 이미지 특징에 더 잘 동작함. 

=> **실시간 애플리케이션에서 빠른 속도와 효율성이 중요한 경우**에 사용.

<br>

### 📌 AKAZE(Accelerated-KAZE)
**[장점]**
- SIFT와 비슷한 성능을 제공하면서도 계산량이 적음.
- 다양한 크기의 특징을 검출하고 크기에 대해 불변성을 가지고 있음.
- 회전, 크기 및 조명 변화에 강함.

**[단점]**
- 매칭 정확도가 SIFT보다는 떨어질 수 있음.
- 대부분의 경우 상대적으로 큰 이미지 특징에 더 잘 동작함.

=> 계산량을 줄이면서도 **일반적인 변형에 대해 강한 성능을 요구하는 경우**에 사용.

<br>


### ===> SIFT가 가장 적합.
<br>
<br>

<h2 id="SIFT_알고리즘">📌 SIFT 알고리즘 </h2>
<br>


* SIFT(Scale-Invariant Feature Transform)이란 ?
> 이미지의 크기와 회전에 불변하는 특징을 추출하는 알고리즘. 

> 서로 다른 두 이미지에서 SIFT 특징을 각각 추출한 다음 서로 가장 비슷한 특징끼리 매칭해주면 두 이미지에서 대응되는 부분을 찾을 수 있다는 것이 기본 원리. 

>크기와 회전은 다르지만 일치하는 내용을 갖고 이미지에서 동일한 물체를 찾아서 매칠해 줄 수 있는 알고리즘.

<br>
 
<br>
<br>

<h2 id="키포인트">02. keypoint / descriptor</h2> 


### 📌 keypoint
- 이미지에서 독특하고 주요한 지점을 나타내는 특징점. 
- 주로 이미지에서 변형에 강한 지점, 경계, 코너 등으로 정의됨.

### 📌 descriptor
- keypoint에 해당하는 정보, 기본적으로 동일한 개수로 생성하며 실제 유사도를 판별하기 위한 데이터로 활용.
- keypoint를 수치적으로 설명하는 역할. 
- keypoint의 주변 영역을 분석하여 특징을 추출하고 벡터 형태로 표현. 
- 이 벡터는 특징점의 주변 정보를 포함하고 있으며 특징점들을 비교하고 매칭하는데 사용됨. 
