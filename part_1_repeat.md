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

        csv["attrib"].value_counts()
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

        train_test_split(X, y, test_size=0.2)
            X_train, X_test, y_train, y_test 분리

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

        차원 축소
            차원 축소를 하면 일부 정보가 유실됨, 따라서 훈련 속도가 빨라질 수 있지만 성능이 감소됨, 파이프라인이 복잡해져 유지보수가 어려워짐
            잡음이나 불필요한 정보를 걸러낼 수 있으므로 성능이 높아질 수도 있음
            보통은 훈련속도만 빨라짐
            시각화에도 유용함
            차원이 클수록 과대적합 위험이 큼

            PCA
                분산을 최대한 보존하는 성분 = 주성분
                첫번째 주성분의 수직 = 두번째, 첫번째와 두번째가 이루는 평면의 수직 = 세번째
                n개의 주성분으로 이루는 n차원으로 축소시킴

                PCA(n_components=2)
                    자동으로 데이터를 중앙으로 정렬
                    PCA.explained_variance_ratio_ = 주성분의 설명된 분산의 비율(80%가 첫번째 PC에 놓여있음, ...)
            
                PCA(n_components=0.95)
                    보존하려는 분산의 비율
                    PCA.n_components_ = 95%에 대한 차원 수 (mnist 154차원)

                PCA.inverse_transform(X_reduced)
                    차원 복원
                    데이터 조금 변경 됨

            랜덤 PCA
                PCA(n_components=0.95, svd_solver="randomized")
                        차원이 샘플수보다 많이 작으면 SVD방식보다 훨씬 빠름

            점진적 PCA
                inc_pca = IncrementalPCA(n_components=154)
                for X_batch in np.array_split(X_train, 100):
                    inc_pca.partial_fit(X_batch)
                X_reduced = inc_pca.transform(X_train)

                memmap이용하여 마찬가지로 메모리 부족 문제 해결 가능

            커널 PCA
                KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
                커널 선택을 해야하기 때문에 gridsearch 이용하여 파라미터 찾기
                KernelPCA, LogisticRegression을 파이프라인으로 만들고 파라미터를 kernelPCA에서 rbf,sigmoid, linear등 설정하여 찾는다

                fit_inverse_transform=True로 하면 재구성 가능(PCA.inverse_transform())
                이후 mean_squared_error(X, X_preimage)하면 재구성 오차 계산
            
            LLE
                LocallyLinearEmbedding(n_components=2, n_neighbors=15)
                    대량의 데이터셋은 힘듬
            
            랜덤 투영, 다차원 스케일링, Isomap, t-SNE, 선형 판별 분석


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
                계단모양으로 결정 경계를 만듬 -> 회전에 민감함 -> 데이터를 좋은 방향으로 회전시키는 PCA기법 사용
            
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
                    회귀 방식으로 클래스를 예측하는 대신 값을 예측함

        앙상블 학습
            투표
                VotingClassifier(
                    estimators=[("model_name", clf), ("model_name", clf), ...],
                    voting=직접투표, 간접투표("hard", "soft", 각 분류기가 확률을 내서 예측)
                )
                    soft방식으로 했을 때 각 분류기가 predict_proba() 메서드가 있어야함(확률 계싼을 할 수 있어야함)
                    모든 분류기가 독립적일수록 최고의 성능을 발휘함

            배깅, 페이스팅
                BaggingClassifier(
                    DecisionTreeClassifier(), n_estimators=생성할 트리의 개수(500),
                    max_samples=100, bootstrap=True, n_jobs=-1
                )
                    bootstrap = True일 경우 배깅, False일 경우 페이스팅
                    배깅 = 훈련 세트를 중복 허용하여 여러개로 나눠서 훈련
                    페이스팅 = 중복 미허용
                    predict_proba있으면 간접 투표 방식 사용
                    편향은 비슷하지만 분산은 줄어듬
                    배깅이 페이스팅보다 편향이 좀 더 높음, 분산 낮음 -> 배깅이 좀더 나은 모델을 만들지만 여유가 있다면 그냥 교차검증

                    배깅을 할 때 훈련 세트를 중복으로 하여 약 63%정도만 사용하여 사용하지 않은 37%정도를 검증세트로 사용함(매 훈련마다 선택되지 않은 샘플은 다름), 테스트 세트와 점수 비슷함
                    -> oob_score=True 하면 oob평가를 사용하여 model.oob_score_으로 불러올수 있음

                    랜덤 패치와 서브스페이스
                        특성 샘플링
                        이미지와 같은 매우 고차원의 데이터셋을 다룰 때 유용
                        모든 샘플과 특성을 샘플링하는걸 랜덤 패치 방식, 특성을 샘플링하는걸 랜덤 서브스페이스 방식
                        랜덤 패치 = bootstrap=False, max_samples=1.0
                        랜덤 서브스페이스 = bootstrap_features=True, max_sample=1.0보다 낮게설정

            랜덤 포레스트
                RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1) (, RandomForestRegressor)
                    DecisionTreeClassifier의 매개변수와 앙상블 자체를 제어하는데 필요한 BaggingClassifier의 매개변수를 모두 가지고 있음
                        (BaggingClassifier(DecisionTreeClassifier(max_features="sqrt, maax_leaf_nodes=16, n_estimators=500))와 유사함)
                
                ExtraTreesClassifier (, ExtraTreesRegressor)
                    익스트림 랜덤 트리, 엑스트라 트리
                    무작위로 특성의 서브셋을 만들어 분할에 사용, 최적의 임곘값을 찾는 대신 후보 특성을 사용해 무작위로 분할한 다음 그중에서 최상의 분할을 선택
                    편향이 늘어나지만 분산을 낮춤
                    일반적으로 랜덤 포레스트보다 엑스트라 트리가 훨씬 빠름 (그냥 교차검증 후 그리드 탐색으로 하이퍼파라미터 튜닝)

                특성 중요도
                    for name, score in zip(data["feature_names"], model.feature_importances_)

            부스팅
                에이다부스트
                    AdaBoostClassifier(
                        DecisionTreeClassifier(max_depth=1), n_estimators=200,
                        algorithm="SAMME.R", learning_rate=0.5
                    )
                        예측을 한 뒤 잘못 분류된 샘플으 가중치를 상대적으로 높인 상대로 다음 분류기에 넘김
                        algorithm의 SAMME은 에이다부스트의 다중 클래스 버전, predict_proba()있으면 확률 추정 가능

                그레디언트 부스팅
                    GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
                        앙상블에 이전까지의 오차를 보정하도록함
                        샘플의 가중치를 추가하는 에이다부스트와 달리 이전 예측기가 만든 잔여 오차에 새로운 예측기를 학습시킴

                        조기 종료 기법 = staged_predict() (p.265 ~ p.266)
                        subsample이용하여 훈련 샘플 비율 정할 수 있음 -> 편향은 높아지고 분산은 낮아짐, 훈련 속도 높임(확률적 그레디언트 부스팅)
                        (-> 외부라이브러리 XGBoost)

                스태킹
                    직접투표같은 예측을 취합하여 결과를 내는것 대신 취합하는 모델을 만드는 것


        KMeans
            KMeans(n_clusters=5)
                KMeans.fit_predict(X) 이용하여 예측
                y_pred == KMeans.labels_
                KMeans.cluster_centers_으로 센트로이드 위치
                KMeans.transform(X)하면 각각 센트로이드까지 거리 확인
                init=(좌표) 로 센트로이드 초기화
                n_init=10 으로 랜덤 초기화 횟수
                KMeans.inertia_로 이너셔 점수 확인

            알고리즘
                랜덤하게 센트로이드를 설정한 후 샘플에 레이블을 할당시킴
                센트로이드를 업데이트 한 후 다시 레이블 할당, 반복
                변하지 않을때까지 반복
                지역 최저점에 멈출 가능성 있음

            MiniBatchKMeans(n_clusters=5)
                미니배치로 훈련

            최적의 클러스터 개수 찾기
                클러스터 개수마다 이너셔 값 그래프를 그려서 급격하게 내려가다가 천천히 내려가게 되는 지점을 엘보라고함, 그 지점이 최적이라고 생각할 수 있음
                -> 믿음직한 방법이 아님 -> 실루엣 점수
                silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]
                좀더 확실하게 실루엣 다이어그램까지 본 후 정하기

                그냥 그리드서치로 구해도 됨
            
            스케일 맞추는게 중요함
            클러스터의 크기가 크거나 밀집도가 다르거나 원형이 아닐경우 잘 작동하지 않음 -> 가우시안 혼합 모델

            준지도 학습
                p.320 - p.322

        DBSCAN
            모든 클러스터가 충분히 밀집되어 있고 밀집되지 않는 지역과 잘 구분될 때 좋은 성능을 냄
            predict() 메서드가 없고 fit_predict()만 제공함 -> 새로운 샘플에 대해 클러스터를 예측할 수 없음

            계산 복잡도
                O(m log m)

            DBSCAN(eos=0.05, min_samples=5)
                DBSCAN.label_에서 일부 값이 -1인 경우는 이상치로 판단함
                핵심 샘플 인덱스 : DBSCAN.core_sample_indices_
                핵심 샘플 자체 : DBSCAN.components_

                KNeighborsClassifier(n_neighbors=50)
                knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

        병합 군집, BIRCH, 평균-이동, 유사도 전파, 스펙트럼 군집

        가우시안 혼합 모델
            GaussianMixture(n_components=3, n_init=10)
                샘플에 활성화, 업데이트 반복함

                gmm.n_iter_ 반복횟수

                하드 군집 gmm.predict()
                소프트 군집 gmm.predict_proba()            

                생성 모델이기 때문에 X_new, y_new = gmm.sample(n)으로 샘플 생성 가능
            
                gmm.score_sample(X)로 각 샘플의 밀도 구하기 

            이상치 탐지
                densities = gmm.score_samples(X)
                density_threshold = np.percentile(densities, 4) (4%)
                anomalies = X[densities < density_threshold]

            클러스터 개수 선택
                BIC, AIC를 최소화하는 모델 선택
                gmm.bic(X), gmm.aic(X)

        베이즈 가우시안 혼합 모델
            BayesianGaussianMixture(n_components=10, n_init=10)
                np.round(bgm.weights_, 2)해서 클러스터가 몇개 필요한지 보면 자동으로 정하는걸 알 수 있음

        이상치 탐지, 특이치 탐지
            PCA(inverse_transform() 메서드를 가진 다른 차원 축소 기법) - 이상치 탐지
            Fast-MCD - 이상치 탐지
            아이솔레이션 포레스트 - 이상치 탐지
            LOF - 이상치 탐지
            one-class SVM - 특이치 탐지


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