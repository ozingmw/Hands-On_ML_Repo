'''
머신러닝 프로젝트 체크리스트
    1. 문제를 정의하고 큰 그림을 그립니다.
    2. 데이터 수집
    3. 데이터 탐색
    4. 데이터 준비
    5. 여러 모델 시험 후 모델 고르기
    6. 모델 튜닝
    7. 솔루션 출시
    8. 출시 및 모니터링, 유지 보수


    데이터 탐색
        csv.head()
            처음 행 표시

        csv.info()
            간략한 설명 및 데이터 타입, 널이 아닌 값 개수

        CSV["attrib"].value_counts()
            해당 특성 값 정보

        csv.describe()
            숫자형 특성 요약 정보

        corr = csv.corr()
        corr["attrib"].sort_values()
            상관관계 조사, 1일수록 증가하면 증가, -1일수록 증가하면 감수, 0일수록 선형적인 관계 없음

        data.DESCR
            data설명

        confusion_matrix(y_train, y_pred)
            오차 행렬
        
        precision_score(y_train, y_pred)
            정밀도

        recall_score(y_train, y_pred)
            재현율

        f1_score(y_train, y_pred)
            f1 score







    데이터 시각화
        csv.hist()
        plt.show()
            히스토그램

        csv.plot(kind="", x="", y="", alpha=0.1)
            산점도(alpha = 투명하게)

        scatter_matrix(csv["attrib1", "attrib2", "attrib3"], figsize=(12,8))
            상관관계 그래프

        conf_mx = confusion_matrix(y_train, y_pred)
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show()
            오차 행렬 시각화, 어두우면 잘 분류 못함
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()
            분류기가 만든 에러 시각화, 밝으면 잘못 분류중

        for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_pred = model.predict(:m)
            y_val_pred= model.predict(X_val)
            train_error.append(mean_squared_error(y_train[:m], y_train_pred))
            val_error.append(mean_squared_error(y_val[:m], y_val_pred))
        plt.plot(np.sqrt(train_error), "r--")
        plt.plot(np.sqrt(val_error), "b--")
            학습 곡선
                두 곡선이 수평한 구간을 이루고 높은 오차에서 가까이 근접 = 과소적합 -> 더 복잡한 모델 사용 or 더 나은 특성 선택(샘플 추가 의미 없음)
                두 곡선이 수평한 상태로 사이에 공간이 있다 = 과대적합 -> 더 큰 훈련 세트(점점 두 선이 가까워 질때까지)
            




    
    데이터 준비
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in sss.split(csv, csv["attrib"]):
            train_idx_s = csv.loc[train_idx]
            test_idx_s = csv.loc[test_idx]
            계층 샘플링 - 비율별로 다르게 샘플링

        imputer = SimpleImputer(strategy="median")
        imputer.fit_transform(csv)
            fit으로 imputer를 조정하고 transform으로 csv를 imputer에 맞게 변경함(strategy="median" - 숫자, "most_frequent" - 숫자, 문자)

        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit_transform(csv)
            문자열 특성 one hot encoding

        StandardScaler()
            표준화

        sklearn.pipline.Pipeline
            파이프라인(묶어서 관리)




    모델
        LinearRegression()
            선형 회귀
                계산 복잡도:
                    O(n) - 샘플 수
                    O(n^2) - 특성 수(SVD방식, sklearn사용)
                    O(n^2.4 ~ n^3) - 특성 수(정규방정식)
                속도 빠름

        SGDRegressor(max_iter=최대 에포크(1000), tol=손실(1e-3), eta0=학습률(0.1), penalty=규제(l2))
            확률적 경사 하강법
                계산 복잡도

                특성이 많고 샘플이 많아도 빠름
                특성 스케일링을 해야 빨라짐
                경사 하강법보단 빠르지만 최적값은 아님, 무작위성으로 인해 지역 최솟값에서 탈출시킴

        PolynomialFeatures(degree=최대차항(2), include_bias=a^2, b^2뿐만 아니라 ab도 포함(True))
            다항 회귀
                p.178
                특성 차수를 늘려 특성을 확장시킴
                코드:
                    poly.fit_transform(X_train)
                    lin_reg.fit(poly, y_train)

        LogisticRegression(C=규제(1, alpha와 역수 클수록 규제 작음), multi_class=소프트맥스 회귀 사용("multinomial", 입력시 사용), solver=알고리즘("lbfgs", 입력시 소프트맥스도 지원))
            로지스틱 회귀(소프트맥스 회귀)
                로지스틱 - > 이진 분류기 / 소프트맥스 -> 다중 이진 분류기
                소프트맥스 회귀는 한번에 하나의 클래스만 예측 -> 한 사진에서 여러사람 얼굴 예측 불가능

        SVM
            서포트 벡터 머신
                복잡한 분류 문제 적합, 작거나 중간 크기의 데이터셋에 적합
                최대한 폭이 넓은 도로 찾기
                스케일링 필요
                하드 마진 분류(이상치에 민감) / 소프트 마진 분류
            
            선형 SVM
                시간 복잡도
                    O(m * n)
                
                LinearSVC(C=폭 넓이(1, 크면 좁아짐 -> 과대적합이면 C감소))
                    선형 SVM모델
                        probability=True 일때 predict_proba()사용 가능
                    (커널중 사용 1번)
                    (SVC(kernel="linear) 보다 LinearSVC가 더 빠름 (특히 훈련 세트가 아주 크거나 특성 수가 많을 때))
                
                SCV(kernel="linear", C=1)
                    선형 커널 사용
                    
                SGDClassifier(alpha=1/(m*C))
                    확률적 경사 하강법 적용
                        LinearSVC보다 느리지만 데이터셋이 크거나 온라인 학습으로 분류할때 사용

            비선형 SVM
                PolynomialFeature(degree=n) 이후 StandardScaler()로 정규화 후 LinearSVC()
                낮은 차수 다항식은 복잡한 데이터셋을 잘 표현하지 못하고 높은 차수 다항식은 모델을 느리게함
                -> 커널 트릭(실제로 특성을 추가하지 않지만 추가한 척)
                
                SVC(kernel="poly", degree=3, coef0=1, C=5)
                    시간 복잡도
                        O(m^2 * n) ~ O(m^3 * n)
                    coef0은 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절,
                    1보다 작은 값과 1보다 큰 값의 차이가 큼
                    과대적합이라면 차수를 줄이기
                
                SVC(kernel="rbf", gamma=5, C=0.001)
                    가우시안 RBF 커널
                        과대적합일 경우 gamma감소, C감소
                        (훈련 세트가 크지 않다면 2번)

            SVM 회귀
                LinearSVR(epsilon=마진(1.5, 도로의 폭))
                    도로안에 최대한 많은 데이터가 들어가게
                    LinearSVC의 회귀버전
                    데이터셋에 비례하여 선형적으로 증가

                SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
                    SVC의 회귀버전
                    데이터셋이 커지면 훨씬 느려짐

            이론
                하드 마진 = 오류 하나도 없이
                소프트 마진 = 제한적인 오류 포함하여 가능한 마진 크게(도로의 폭)
                둘다 선형적인 제약 조건이 있어 콰드라틱 프로그래밍문제
                쌍대 문제



            결정 트리
                DecisionTreeClassifier(max_depth=결정 경계(2))
                    훈련시키기 위해 CART알고리즘 사용, 그리디 알고리즘, NP-완전 문제로 최적의 트리 찾기 어려움
                    시간 복잡도
                        O(log2(m)) -> 특성 수와 무관
                    훈련 복잡도
                        O(n * m log2(m))
                        훈련 세트가 수천개 이하 정도로 작을 경우 presort=True 하면 빨라짐
                    지니 불순도 or 엔트로피
                        criterion 기본 = 지니 불순도, "entropy" = 엔트로피
                            큰 차이 없음, 지니 불순도가 조금 더 빠름, 엔트로피가 조금 더 균형잡힘
                    max_depth 줄이면 과대적합 감소
                    
                    매개변수
                        min_sample_split - 분할되기 위해 노드가 가져야 하는 최소 샘플 수
                        min_sample_leaf - 리프 노드가 가지고 있어야 할 최소 샘플 수
                        min_weight_fraction_leaf - min_sample_leaf와 같지만 가중치가 부여된 전체 샘플 수에서의 비율
                        max_leaf_nodes - 리프 노드 최대 수
                        max_features - 각 노드에서 분할에 사용할 특성의 최대 수

                        min시작하는 매개변수 증가, max시작하는 매개변수 감소시키면 규제가 커짐
                
                DecisionTreeRegressor()






    모델 규제
        Ridge(alpha=규제(1e-3, 0이면 선형회귀와 같아짐, 크면 평균을 지나는 수평선처럼 됨), solver=계산 방법("cholesky", 정규방정식))
        SGDRegressor(penalty="l2")
            릿지 규제
                전역 최적점에 가까워질수록 그레디언트가 작아짐
        Lasso(alpha=규제(0.1))
        SGDRegressor(penalty="l1")
            라쏘 규제
                덜 중요한 특성의 가중치를 제거
                SGD사용하여 최적점 근처에서 진동하는걸 막을거면 학습률을 감소시켜야함
        ElasticNet(alpha=규제(0.1), l1_ratio=비율(0.5))
            엘라스틱넷
                릿지와 라쏘 절충안
                r(l1_ratio)로 조절(릿지 회귀=0, 라쏘 회귀=1)
        SGDRegressor(max_iter=1, warm_start=True)
            조기 종료
                warm_start=True로 하면 fit메서드가 호출될 때 이전 모델 파라미터에서 훈련시작
                매 훈련 순간 검증 오차를 구해서 검증오차가 줄어들지 않으면 강제 종료시킴




    모델 튜닝
        GridSearchCV(model, params, cv=5, scoring="")
            그리드 탐색
        
        RandomizedSearchCV(model, params, cv=5)
            랜덤 탐색

                








    모델 평가
        mse = mean_squared_error(y_test, y_pred)
        np.sqrt(mse)
            rmse, 낮을수록 좋음

        score = cross_val_score(model, X_test, y_test, scoreing="", cv=10)
        np.sqrt(-score)
            k겹 교차 검증





'''