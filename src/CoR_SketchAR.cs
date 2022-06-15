using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.CoreModule;
using System.Collections.Generic;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using OpenCVMarkerLessAR;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using OpenCVForUnity.PhotoModule;
using System;

namespace MarkerLessARExample
{
    public class ShapeDetectionWithCrowd : MonoBehaviour
    {
        [Header("------------------ OpenCV")]
        [Space(5f)]

        public WebCamTextureToMatHelper webCamTextureToMatHelper;

        Texture2D texture;

        Mat grayMat;
        Mat filter;
        Mat fftimg;
        List<Mat> hlsimg;
        Mat hls0;

        Mat rgbaMat;

        Mat rgbMat;

        Mat hsvMat;
        Mat hlsMat;
        Mat grayMatForHough;

        Mat camMatrix;

        MatOfDouble distCoeffs;

        Matrix4x4 invertYM;

        Matrix4x4 invertZM;

        public Camera ARCamera;

        public bool shouldMoveARCamera;

        public int redMin, redMax;
        public int blueMin, blueMax;
        public int greenMin, greenMax;
        public int pinkMin, pinkMax;
        public int saturationMin, valueMin;
        public float rectMin,planeMin;
        public float isSameRange;
        public float timeBeforeObst;

        GameObject crossWalkObj;

        [Header("------------------ Status")]
        [Space(5f)]
        public bool pointMap = false;
        public bool enableLowPassFilter;
        public bool isObstMapping = false;
        public bool isSame = false;
        [Space(10f)]

        [Header("------------------ ARObject Prefab")]
        [Space(5f)]
        public GameObject ARGameObject;
        public GameObject ARGameObjectBlue;
        public GameObject ARGameObjectGreen;
        public GameObject ARGameObjectCrossWalk;
        public GameObject ARGameObjectPlane;
        public GameObject roofPrefab;
        public GameObject ARGameObjectObst;
        public GameObject ARGameObjectObst2;
        public GameObject ARGameObjectObst3;
        public GameObject ARGameObjectObst4;
        public GameObject ARGameObjectTree;
        [Space(10f)]

        List<GameObject> buildingObj;
        List<GameObject> building3DObj;
        List<GameObject> roadObj;
        List<GameObject> treeObj;
        List<GameObject> tree3DObj;
        GameObject planeObj;
        GameObject plane3DObj;

        [Header("------------------ Material")]
        [Space(5f)]
        public Material Transparency;
        public Material roadTexture3D;
        public Material crosswalkTexture3D;
        [Space(10f)]

        Mat imgMat;
        Mat srcHierarchy;

        //color Idx
        int colorIdx = 0;

        [Header("------------------ Matrix")]
        [Space(5f)]
        public Matrix4x4 pPose3d;
        public Matrix4x4 pose3d;
        public List<Matrix4x4> transformBuildM;
        public Matrix4x4 treePose3d;
        public List<Matrix4x4> transformTreeM;
        [Space(10f)]

        public MatOfPoint3f points3d;
        public MatOfPoint2f points2d;


        public Toggle enableLowPassFilterToggle;
        public float positionLowPass = 0.005f;
        public float rotationLowPass = 2f;
        PoseData poseData;
        PoseData oldPoseData;
        Matrix4x4 ARM;

        //외곽선 검출
        List<MatOfPoint> tempContours = new List<MatOfPoint>();
        List<MatOfPoint> srcContours = new List<MatOfPoint>();
        List<Point> srcContoursCenter = new List<Point>();
        List<MatOfPoint> buildContours = new List<MatOfPoint>();
        List<Point> preBuildContours= new List<Point>();
        List<Point> buildContoursCenter = new List<Point>();
        List<MatOfPoint> roadContours = new List<MatOfPoint>();
        List<Point> roadContoursCenter = new List<Point>();
        List<MatOfPoint> treeContours = new List<MatOfPoint>();
        List<Point> preTreeContours = new List<Point>();
        List<Point> treeContoursCenter = new List<Point>();
        List<MatOfPoint> triContours = new List<MatOfPoint>();
        public List<Point> wayPointConvexContours = new List<Point>();
        public List<Point> wayPointContours = new List<Point>();
        List<float> wayPDist = new List<float>();

        MatOfPoint planeContour = new MatOfPoint();
        Point prePlaneContour = new Point();
        Point planeContourCenter = new Point();
        List<Point3> pPoints3dList = new List<Point3>(4);
        MatOfPoint3f pPoint3d = new MatOfPoint3f();

        List<Point3> points3dList = new List<Point3>(4);
        MatOfPoint3f point3d = new MatOfPoint3f();
        public List<MatOfPoint3f> buildPoint3d = new List<MatOfPoint3f>();


        List<Point3> treePoints3dList = new List<Point3>(4);
        MatOfPoint3f tPoint3d = new MatOfPoint3f();
        public List<MatOfPoint3f> treePoint3d = new List<MatOfPoint3f>();

        Mat raux = new Mat();
        Mat taux = new Mat();
        Point point = new Point();
        float[] radius = new float[1];

        //잔디 모형들
        public GameObject[] grassObj;

        Mat circles;
        Point pt;
        public Point startPt;
        public Point endPt;
        Point distX;

        Mat exitCircles;
        public List<Point> obstExitUp;
        public List<Point> obstExitDown;

        Mat obstButtonCircle;

        [Header("------------------ CrowdSimulation")]
        [Space(5f)]
        public CrowdSimulationManager crowdManager;
        public float crowdSize = 20;
        public bool isInterpolation = false;
        public bool isStartEndPoint = false;
        Mat lines;

        public int boundH, boundS, boundV;
        MatOfInt4 convexityDegects = new MatOfInt4();
        Point[] cnt_arr;
        MatOfInt hull;

        public Material quadRed;
        public Material quadBrown;

        IList<RVO.Vector2> vertex = new List<RVO.Vector2>();
        IList<RVO.Vector2> vertexTop;
        IList<RVO.Vector2> vertexBottom ;
        IList<RVO.Vector2> vertexLeft ;
        IList<RVO.Vector2> vertexRight;

        public Text obstMappingTxt;

        public Text statusTxt;

        float timer;        //업데이트 하는 속도

        public float moveSpeed;


        public Vector3 prevBuildPos;
        public List<Vector3> prevTreePos;

        private Vector3 staticPos;
        private Vector3 staticScale;

        private float exitThickness = 0.7f;

        Dictionary<int, int> idx = new Dictionary<int, int>();
        bool[] checkExcept = new bool[3] { false,false,false };
        int exceptBuildNum;

        Vector3[] vertices;
        Vector3[] vertice;
        Vector3[] c;
        int[] triangles;
        Vector2[] uvs;
        Mesh cubeM;

        // Use this for initialization
        void Start()
        {
            imgMat = new Mat(640,480, CvType.CV_8UC4);

            float width = imgMat.width();
            float height = imgMat.height();

            float imageSizeScale = 1.0f;
            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
                imageSizeScale = (float)Screen.height / (float)Screen.width;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }

            crowdSize = 20;

            points2d = new MatOfPoint2f();
            points3d = new MatOfPoint3f();
            pose3d = new Matrix4x4();
            pPose3d = new Matrix4x4();

            // Build 2d and 3d contours (3d contour lie in XY plane since it's planar)
            List<Point> points2dList = new List<Point>(4);

            float w = 480.0f;
            float h = 480.0f;

            points2dList.Add(new Point(0, 0));
            points2dList.Add(new Point(w, 0));
            points2dList.Add(new Point(w, h));
            points2dList.Add(new Point(0, h));

            points2d.fromList(points2dList);

            //초기화
            buildingObj = new List<GameObject>();
            building3DObj = new List<GameObject>();
            roadObj = new List<GameObject>();
            treeObj = new List<GameObject>();
            tree3DObj = new List<GameObject>();

            points3dList = new List<Point3>(4);
            point3d = new MatOfPoint3f();
            buildPoint3d = new List<MatOfPoint3f>();

            treePoints3dList = new List<Point3>(4);
            tPoint3d = new MatOfPoint3f();
            treePoint3d = new List<MatOfPoint3f>();

            planeContour = new MatOfPoint();
            prePlaneContour = new Point();
            planeContourCenter = new Point();
            pPoints3dList = new List<Point3>(4);
            pPoint3d = new MatOfPoint3f();

            hsvMat = new Mat();
            hlsMat = new Mat();
            grayMat = new Mat();
            fftimg = new Mat();
            filter = new Mat();
            hlsimg = new List<Mat>();

            grayMatForHough = new Mat();
            rgbMat = new Mat();

            srcHierarchy = new Mat();

            hull = new MatOfInt();
            convexityDegects = new MatOfInt4();

            raux = new Mat();
            taux = new Mat();

            point = new Point();
            radius = new float[1];

            prevBuildPos = new Vector3();
            prevTreePos = new List<Vector3>();

            staticPos = new Vector3(320, 240, -640);
            staticScale = new Vector3(20, 20, 20);

            //외곽선 검출
            tempContours = new List<MatOfPoint>();
            srcContours = new List<MatOfPoint>();

            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();

            webCamTextureToMatHelper.Initialize();

            wayPointConvexContours = new List<Point>();
            wayPointContours = new List<Point>();
            wayPDist = new List<float>();

            if (SystemInfo.supportsGyroscope)
            {
                Input.gyro.enabled = true;
            }

            isInterpolation = false;
            isStartEndPoint = false;
            isObstMapping = false;

            if (Application.platform == RuntimePlatform.WindowsEditor)
            {
                settingColorRange(128, 180, 101, 120, 80, 100,140,170, 30, 130);
                rectMin = 150.0f;
                planeMin = 15000.0f;               //최대
                timeBeforeObst = 1.0f;
                isSameRange = rectMin / 4;
            }
            else if (Application.platform == RuntimePlatform.Android)
            {
                settingColorRange(0, 60, 81, 180, 61, 80,160,169, 20, 200);
                rectMin = 100.0f;
                planeMin = 12000.0f;               //최대
                timeBeforeObst = 0.7f;
                isSameRange = rectMin / 4;
            }

            triangles = new int[]
                {
                        3, 1, 0,        3, 2, 1,        // Bottom	
                    7, 5, 4,        7, 6, 5,        // Left
                    11, 9, 8,       11, 10, 9,      // Front
                    15, 13, 12,     15, 14, 13,     // Back
                    19, 17, 16,     19, 18, 17,	    // Right
                    23, 21, 20,     23, 22, 21,     // Top
                };


            Vector2 uv00 = new Vector2(0f, 0f);
            Vector2 uv10 = new Vector2(1f, 0f);
            Vector2 uv01 = new Vector2(0f, 1f);
            Vector2 uv11 = new Vector2(1f, 1f);

            uvs = new Vector2[]
            {
                uv11, uv01, uv00, uv10, // Bottom
	            uv11, uv01, uv00, uv10, // Left
	            uv11, uv01, uv00, uv10, // Front
	            uv11, uv01, uv00, uv10, // Back	        
	            uv11, uv01, uv00, uv10, // Right 
	            uv11, uv01, uv00, uv10  // Top
            };
        }

        public void settingColorRange(int redMinV,int redMaxV,int blueMinV,int blueMaxV,int greenMinV,int greenMaxV,int pinkMinV,int pinkMaxV,int saturationMinV,int valueMinV)
        {
            redMin = redMinV;
            redMax = redMaxV;
            blueMin = blueMinV;
            blueMax = blueMaxV;
            greenMin = greenMinV;
            greenMax = greenMaxV;
            pinkMin = pinkMinV;
            pinkMax = pinkMaxV;
            saturationMin = saturationMinV;
            valueMin = valueMinV;
        }

        /// <summary>
        /// Raises the web cam texture to mat helper initialized event.
        /// </summary>
        public void OnWebCamTextureToMatHelperInitialized()
        {
            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

            texture = new Texture2D(webCamTextureMat.width(), webCamTextureMat.height(), TextureFormat.RGBA32, false);
            gameObject.GetComponent<Renderer>().material.mainTexture = texture;

            grayMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC1);

            gameObject.transform.localScale = new Vector3(webCamTextureMat.width(), webCamTextureMat.height(), 1);

            float width = webCamTextureMat.width();
            float height = webCamTextureMat.height();

            float imageSizeScale = 1.0f;
            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
                imageSizeScale = (float)Screen.height / (float)Screen.width;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }

            //set cameraparam
            int max_d = (int)Mathf.Max(width, height);
            double fx = max_d;
            double fy = max_d;
            double cx = width / 2.0f;
            double cy = height / 2.0f;
            camMatrix = new Mat(3, 3, CvType.CV_64FC1);
            camMatrix.put(0, 0, fx);
            camMatrix.put(0, 1, 0);
            camMatrix.put(0, 2, cx);
            camMatrix.put(1, 0, 0);
            camMatrix.put(1, 1, fy);
            camMatrix.put(1, 2, cy);
            camMatrix.put(2, 0, 0);
            camMatrix.put(2, 1, 0);
            camMatrix.put(2, 2, 1.0f);

            distCoeffs = new MatOfDouble(0, 0, 0, 0);

            //calibration camera
            Size imageSize = new Size(width * imageSizeScale, height * imageSizeScale);
            double apertureWidth = 0;
            double apertureHeight = 0;
            double[] fovx = new double[1];
            double[] fovy = new double[1];
            double[] focalLength = new double[1];
            Point principalPoint = new Point(0, 0);
            double[] aspectratio = new double[1];

            Calib3d.calibrationMatrixValues(camMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectratio);

            //To convert the difference of the FOV value of the OpenCV and Unity. 
            double fovXScale = (2.0 * Mathf.Atan((float)(imageSize.width / (2.0 * fx)))) / (Mathf.Atan2((float)cx, (float)fx) + Mathf.Atan2((float)(imageSize.width - cx), (float)fx));
            double fovYScale = (2.0 * Mathf.Atan((float)(imageSize.height / (2.0 * fy)))) / (Mathf.Atan2((float)cy, (float)fy) + Mathf.Atan2((float)(imageSize.height - cy), (float)fy));

            //Adjust Unity Camera FOV https://github.com/opencv/opencv/commit/8ed1945ccd52501f5ab22bdec6aa1f91f1e2cfd4
            if (widthScale < heightScale)
            {
                ARCamera.fieldOfView = (float)(fovx[0] * fovXScale);
            }
            else
            {
                ARCamera.fieldOfView = (float)(fovy[0] * fovYScale);
            }

            invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));
            invertZM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, 1, -1));

            //if WebCamera is frontFaceing,flip Mat.
            webCamTextureToMatHelper.flipHorizontal = webCamTextureToMatHelper.GetWebCamDevice().isFrontFacing;
        }

        // Update is called once per frame
        void Update()
        {
           
            if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                rgbaMat = webCamTextureToMatHelper.GetMat();

                Size size = new Size(10.0f, 10.0f);
                CLAHE clahe = Imgproc.createCLAHE(5.0, size);

                Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY);                            //rgbaMat를 gray로 바꿔서 grayMat에 저장
                Imgproc.cvtColor(rgbaMat, hsvMat, Imgproc.COLOR_RGBA2RGB);                                   //rgbaMat를 hsv로 바꿔서 hsvMat에 저장
                Imgproc.cvtColor(hsvMat, hsvMat, Imgproc.COLOR_RGB2HSV);                                   //rgbaMat를 hsv로 바꿔서 hsvMat에 저장

                clahe.apply(grayMat, grayMat);                                                                                          //clahe 함수를 통해 grayMat를 명암비를 평탄화함

                Imgproc.GaussianBlur(grayMat, grayMat, new Size(0, 0), 1);

                // grap only the Y component.
                Core.extractChannel(grayMat, grayMat, 0);

                Imgproc.Canny(grayMat, grayMat, 50,200);

                Imgproc.cvtColor(rgbMat, grayMatForHough, Imgproc.COLOR_RGB2GRAY);
                Imgproc.Canny(grayMatForHough, grayMatForHough, 50, 200);

                timer += Time.deltaTime;

                float reloadTime = (isObstMapping) ? 0.1f : timeBeforeObst;

                if (timer > reloadTime)
                {
                    if (!isObstMapping)
                    {
                        allDestroy();           //모두 지우는 것이 아닌 비교한 후 달라졌으면 다시 지우고 그릴 것(수정 필요)
                    }
                    timer = 0.0f;
                    tempContours.Clear();
                }

                isSame = false;


                if (!isObstMapping)
                {
                    //점들 찾기
                    Imgproc.findContours(grayMat, tempContours, srcHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                }
                else
                {
                    Imgproc.findContours(grayMat, tempContours, srcHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                }


                //찾은 점들을 가지고 사각형 찾기
                foreach (var cnt in tempContours)
                {
                    Imgproc.convexHull(cnt, hull, false);

                    cnt_arr = cnt.toArray();
                    int[] hull_arr = hull.toArray();
                    Point[] pts = new Point[hull_arr.Length];
                    for (int i = 0; i < hull_arr.Length; i++)
                    {
                        pts[i] = cnt_arr[hull_arr[i]];
                    }

                    MatOfPoint2f ptsFC2 = new MatOfPoint2f(pts);
                    MatOfPoint2f approxFC2 = new MatOfPoint2f();
                    MatOfPoint approxSC2 = new MatOfPoint();

                    double arclen = Imgproc.arcLength(ptsFC2, true);
                    Imgproc.approxPolyDP(ptsFC2, approxFC2, 0.02 * arclen, true);
                    approxFC2.convertTo(approxSC2, CvType.CV_32S);

                    if (approxSC2.size().area() != 4)          //사각형이지 않을때 스킵
                        continue;

                    addObstacleFunc(approxSC2);                                //obstacle 생성

                    if (!isObstMapping)
                    {
                        UpdateObstalceTransform();                                //update 위치 변경
                    }
                }

                //빌딩 관련
                if (isObstMapping)
                {
                    //plane을 기반으로 움직이기
                    pPose3d = new Matrix4x4();
                    computeBuildPose(pPoint3d, new MatOfPoint2f(planeContour.toArray()), camMatrix, distCoeffs, 1);
                    Matrix4x4 ARMPlane = invertYM * pPose3d * invertYM;
                    //Apply Y-axis and Z-axis refletion matrix. (Adjust the posture of the AR object)
                    ARMPlane = ARMPlane * invertYM * invertZM;
                    ARMPlane = ARCamera.transform.localToWorldMatrix * ARMPlane;
                    ARUtils.SetTransformFromMatrix(plane3DObj.transform, ref ARMPlane);
                }

                //만약 군중 시뮬레이션이 시작되지 않았다면 출력
                if (!crowdManager.isStart)
                {
                    if (wayPointConvexContours.Count > 0)
                    {
                        for (int i = 0; i < wayPointContours.Count; i++)
                            Imgproc.circle(rgbaMat, wayPointContours[i], 5, new Scalar(0, 125, 0, 255), -1);           //원그리기
                        for (int i = 0; i < wayPointConvexContours.Count; i++)
                            Imgproc.circle(rgbaMat, wayPointConvexContours[i], 5, new Scalar(0, 255, 0, 255), -1);           //원그리기
                    }
                    if (isInterpolation && startPt != null && endPt != null)
                    {
                        //시작지점과 끝지점 사이의 벡터를 구하고 표시하기
                        Imgproc.line(rgbaMat, startPt, endPt, new Scalar(255, 0, 100, 255));
                    }

                    if (circles != null)
                    {
                        for (int j = 0; j < circles.cols(); j++)
                        {
                            double[] data = circles.get(0, j);
                            double rho = data[2];
                            if (startPt != null)
                                Imgproc.circle(rgbaMat, startPt, (int)rho, new Scalar(255, 0, 0, 255), 5);
                            if (endPt != null)
                                Imgproc.circle(rgbaMat, endPt, (int)rho, new Scalar(255, 0, 255, 255), 5);
                        }
                    }

                    if (exitCircles != null)
                    {
                        if (exitCircles.cols() > 0)
                        {
                            if (obstExitDown.Count == obstExitUp.Count)
                            {
                                for (int i = 0; i < obstExitUp.Count; i++)
                                {
                                    double[] data = exitCircles.get(0, 0);
                                    double rho = data[2];
                                    if (obstExitUp != null)
                                    {
                                        Imgproc.circle(rgbaMat, obstExitUp[i], (int)rho, new Scalar(0, 255, 0, 255), 5);
                                    }
                                    if (obstExitDown != null)
                                    {
                                        Imgproc.circle(rgbaMat, obstExitDown[i], (int)rho, new Scalar(0, 255, 255, 255), 5);
                                    }
                                }
                            }
                        }
                    }
                }

                Utils.fastMatToTexture2D(rgbaMat, texture);
            }
        }

        public void settingInitialMap()
        {
            if (!pointMap)
            {
                for (int i = 0; i < buildContours.Count; i++)
                {
                    points3dList.Clear();
                    point3d = new MatOfPoint3f();
                    Point[] point2 = buildContours[i].toArray();
                    Point cPoint = buildContoursCenter[i];

                    points3dList.Add(-new Point3(point2[0].x - cPoint.x, point2[0].y - cPoint.y, 0));
                    points3dList.Add(-new Point3(point2[1].x - cPoint.x, point2[1].y - cPoint.y, 0));
                    points3dList.Add(-new Point3(point2[2].x - cPoint.x, point2[2].y - cPoint.y, 0));
                    points3dList.Add(-new Point3(point2[3].x - cPoint.x, point2[3].y - cPoint.y, 0));

                    point3d.fromList(points3dList);
                    buildPoint3d.Add(point3d);
                }

                for (int i = 0; i < treeContours.Count; i++)
                {
                    treePoints3dList.Clear();
                    tPoint3d = new MatOfPoint3f();
                    Point[] point3 = treeContours[i].toArray();
                    Point tPoint = treeContoursCenter[i];

                    treePoints3dList.Add(-new Point3(point3[0].x - tPoint.x, point3[0].y - tPoint.y, 0));
                    treePoints3dList.Add(-new Point3(point3[1].x - tPoint.x, point3[1].y - tPoint.y, 0));
                    treePoints3dList.Add(-new Point3(point3[2].x - tPoint.x, point3[2].y - tPoint.y, 0));
                    treePoints3dList.Add(-new Point3(point3[3].x - tPoint.x, point3[3].y - tPoint.y, 0));

                    tPoint3d.fromList(treePoints3dList);
                    treePoint3d.Add(tPoint3d);
                }

                //plane
                pPoints3dList.Clear();
                pPoint3d = new MatOfPoint3f();
                Point[] point4 = planeContour.toArray();
                Point pPoint = planeContourCenter;
                pPoints3dList.Add(-new Point3(point4[0].x - pPoint.x, point4[0].y - pPoint.y, 0));
                pPoints3dList.Add(-new Point3(point4[1].x - pPoint.x, point4[1].y - pPoint.y, 0));
                pPoints3dList.Add(-new Point3(point4[2].x - pPoint.x, point4[2].y - pPoint.y, 0));
                pPoints3dList.Add(-new Point3(point4[3].x - pPoint.x, point4[3].y - pPoint.y, 0));
                pPoint3d.fromList(pPoints3dList);


                //빌딩 위치 바꾸는 부분
                for (int i = 0; i < buildContours.Count; i++)
                {
                    computeBuildPose(buildPoint3d[i], new MatOfPoint2f(buildContours[i].toArray()), camMatrix, distCoeffs, 0);
                    ARM = invertYM * pose3d * invertYM;
                    // Apply Y-axis and Z-axis refletion matrix. (Adjust the posture of the AR object)
                    ARM = ARM * invertYM * invertZM;
                    ARM = ARCamera.transform.localToWorldMatrix * ARM;
                    ARUtils.SetTransformFromMatrix(building3DObj[i].transform, ref ARM);
                }

                //모든 obj들 모두 해당 rotation 적용   //나무부분
                for (int i = 0; i < treeContours.Count; i++)
                {
                    computeBuildPose(buildPoint3d[i], new MatOfPoint2f(buildContours[i].toArray()), camMatrix, distCoeffs, 2);
                    ARM = invertYM * treePose3d * invertYM;
                    // Apply Y-axis and Z-axis refletion matrix. (Adjust the posture of the AR object)
                    ARM = ARM * invertYM * invertZM;
                    ARM = ARCamera.transform.localToWorldMatrix * ARM;
                    ARUtils.SetTransformFromMatrix(tree3DObj[i].transform, ref ARM);
                }

                //plane을 기반으로 움직이기
                computeBuildPose(pPoint3d, new MatOfPoint2f(planeContour.toArray()), camMatrix, distCoeffs, 1);
                ARM = invertYM * pPose3d * invertYM;
                // Apply Y-axis and Z-axis refletion matrix. (Adjust the posture of the AR object)
                ARM = ARM * invertYM * invertZM;
                ARM = ARCamera.transform.localToWorldMatrix * ARM;
                ARUtils.SetTransformFromMatrix(plane3DObj.transform, ref ARM);

                //object들 부모를 plane으로 변경
                for (int i = 0; i < building3DObj.Count; i++)
                {
                    building3DObj[i].transform.SetParent(plane3DObj.transform, true);
                    changeQuadToCube(buildingObj[i], 10.0f);
                    buildingObj[i].transform.SetParent(plane3DObj.transform, true);
                }
                for (int i = 0; i < tree3DObj.Count; i++)
                {
                    tree3DObj[i].transform.SetParent(plane3DObj.transform, true);
                    changeQuadToCube(treeObj[i], 10.0f);
                    treeObj[i].transform.SetParent(plane3DObj.transform, true);
                }
                for (int i = 0; i < roadObj.Count; i++)
                {
                    roadObj[i].transform.SetParent(plane3DObj.transform, true);
                    changeQuadToCube(roadObj[i], 30.0f);
                    roadObj[i].transform.GetChild(0).GetComponent<MeshRenderer>().material = roadTexture3D;
                }
                if (crossWalkObj != null)
                {
                    crossWalkObj.transform.SetParent(plane3DObj.transform, true);
                    changeQuadToCube(crossWalkObj, 30.0f);
                    crossWalkObj.transform.GetChild(0).GetComponent<MeshRenderer>().material = crosswalkTexture3D;
                }

                crowdManager.CrowdParent.transform.localScale = staticScale;
                crowdManager.CrowdParent.transform.SetParent(plane3DObj.transform, true);

                plane3DObj.GetComponent<DelayableSetActive>().SetActive(true);

                pointMap = true;
            }
        }

        //장애물(나무,건물,도로) 추가하는 함수
        public void addObstacleFunc(MatOfPoint approxSC2)
        {
            //rgbaMat
            double[] colorArray = new double[3];
            double[] colorArr;

            //컬러의 평균값 계산
            for (int i = 0; i < 4; i++)
            {
                double color = approxSC2.toList()[i].x;
                double color2 = approxSC2.toList()[i].y;
                colorArr = hsvMat.get((int)color2, (int)color);
                if ((colorArray[0] / (i + 1)) - colorArr[0] < 10)
                {
                    colorArray[0] += colorArr[0];
                    colorArray[1] += colorArr[1];
                    colorArray[2] += colorArr[2];
                }
            }

            for (int i = 0; i < 3; i++)
                colorArray[i] /= 4;

            colorArray[0] = Math.Truncate(colorArray[0]);

            Point point = new Point();
            float[] radius = new float[1];
            Imgproc.minEnclosingCircle(new MatOfPoint2f(approxSC2.toArray()), point, radius);

            //빌딩 순서대로 sorting 한다음 add하기 순서대로 정렬하기
            Point[] buildPoints = approxSC2.toArray();
            buildPoints = sortingRectPoints(buildPoints);
           
            //도로 순서대로 sorting 한 다음 add하기 순서대로 정렬하기
            if (colorArray[1] > saturationMin&& colorArray[2] > valueMin)               //최소한의 검정색이 아닐때
            {
                //같은게 있는지 확인 검사 필요
                isSame = false;

                if (srcContoursCenter.Count > 0)
                {
                    for (int i = 0; i < srcContoursCenter.Count; i++)
                    {
                        if ((point.x >= srcContoursCenter[i].x - isSameRange && point.x <= srcContoursCenter[i].x + isSameRange)
                            && (point.y >= srcContoursCenter[i].y - isSameRange && point.y <= srcContoursCenter[i].y + isSameRange))
                        {
                            isSame = true;
                        }
                        else
                        {
                            isSame = false;
                        }
                        if (isSame) break;
                    }
                }

                if (!isObstMapping&&!isSame)
                {
                    srcContoursCenter.Add(point);

                    //초록색
                    if ((colorArray[0] >= greenMin && colorArray[0] <= greenMax))
                    {
                        srcContours.Add(approxSC2);
                        treeContours.Add(approxSC2);
                        treeContoursCenter.Add(point);
                        if (new MatOfPoint2f(approxSC2.toArray()).get(3, 0) != null)
                        {
                            //새로운 object 나무 생성하고 위치값 할당
                            treeObj.Add(Instantiate(ARGameObjectGreen, new Vector3(320, 240, -640), Quaternion.Euler(new Vector3(0, 0, 0))));
                        }
                    }
                    else if ((colorArray[0] >= redMin && colorArray[0] <= redMax))       //빨강
                    {
                        approxSC2.fromArray(buildPoints);

                        srcContours.Add(approxSC2);
                        buildContours.Add(approxSC2);
                        buildContoursCenter.Add(point);
                        if (new MatOfPoint2f(approxSC2.toArray()).get(3, 0) != null)
                        {
                            //새로운 object 생성하고 위치값 할당
                            buildingObj.Add(Instantiate(ARGameObject, new Vector3(320, 240, -640), Quaternion.Euler(new Vector3(0, 0, 0))));
                        }
                    }
                    else if ((colorArray[0] >= blueMin && colorArray[0] <= blueMax))     //파랑
                    {
                        approxSC2.fromArray(buildPoints);

                        srcContours.Add(approxSC2);
                        roadContours.Add(approxSC2);
                        roadContoursCenter.Add(point);
                        if (new MatOfPoint2f(approxSC2.toArray()).get(3, 0) != null)
                        {
                            //새로운 object 생성하고 위치값 할당
                            roadObj.Add(Instantiate(ARGameObjectBlue, new Vector3(320, 240, -640), Quaternion.Euler(new Vector3(0, 0, 0))));

                            //만약 도로가 2개 이상일때 횡단보도 설치
                            if (roadObj.Count > 1)
                            {
                                //횡단보도 설치를 위한 2쌍의 경우 체크
                                MatOfPoint2f src = new MatOfPoint2f(roadContours[0].toArray());

                                double[] dVert = src.get(0, 0);
                                double[] dVert4 = src.get(3, 0);

                                MatOfPoint2f src2 = new MatOfPoint2f(roadContours[1].toArray());

                                double[] src2dVert2 = src2.get(1, 0);
                                double[] src2dVert3 = src2.get(2, 0);

                                Point[] roadPoints = new Point[4];

                                if (roadContoursCenter[0].y > roadContoursCenter[1].y)
                                {
                                    //0이 위인 경우
                                    roadPoints[0] = new Point(src2dVert3);
                                    roadPoints[1] = new Point(dVert);
                                    roadPoints[2] = new Point(dVert4);
                                    roadPoints[3] = new Point(src2dVert2);
                                }
                                else{
                                    Debug.LogError("roadContoursCenter Not Sorting");
                                }

                                if (crossWalkObj == null)
                                {
                                    //횡단보도 제작하기
                                    crossWalkObj = Instantiate(ARGameObjectCrossWalk, new Vector3(320, 240, -640), Quaternion.Euler(new Vector3(0, 0, 0)));

                                    MeshFilter cubeMesh = crossWalkObj.transform.GetChild(0).GetComponent<MeshFilter>();

                                    Vector3[] vertice = cubeMesh.mesh.vertices;

                                    vertice[0] = -new Vector3((float)roadPoints[1].x, (float)roadPoints[1].y, 0.1f);          //dVert
                                    vertice[1] = -new Vector3((float)roadPoints[3].x, (float)roadPoints[3].y, 0.1f);        //src2dVert2
                                    vertice[2] = -new Vector3((float)roadPoints[2].x, (float)roadPoints[2].y, 0.1f);        //dVert4
                                    vertice[3] = -new Vector3((float)roadPoints[0].x, (float)roadPoints[0].y, 0.1f);       //src2dVert3

                                    cubeMesh.mesh.vertices = vertice;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                if (Imgproc.contourArea(approxSC2) > planeMin)
                {        
                    approxSC2.fromArray(buildPoints);
                    //검정색 네모인경우
                    planeContour = new MatOfPoint(approxSC2);
                    planeContourCenter = point.clone();
                    if (planeObj == null)
                    {
                        if (new MatOfPoint2f(approxSC2.toArray()).get(3, 0) != null)
                        {
                            planeObj = Instantiate(ARGameObjectPlane, new Vector3(320, 240, -640), Quaternion.Euler(new Vector3(0, 0, 0)));
                            changeVerticesOfCube(planeObj.transform.GetChild(0).GetComponent<MeshFilter>(), new MatOfPoint2f(approxSC2.toArray()));
                        }
                    }
                    else
                    {
                        Calib3d.solvePnP(pPoint3d, new MatOfPoint2f(approxSC2.toArray()), camMatrix, distCoeffs, raux, taux);
                        changeVerticesOfCube(planeObj.transform.GetChild(0).GetComponent<MeshFilter>(), new MatOfPoint2f(approxSC2.toArray()));
                    }
                }
            }
        }

        private Point[] sortingRectPoints(Point[] buildPoints)
        {
            for (int i = 0; i < buildPoints.Length; i++)
            {
                //x순으로 정렬하기
                for (int j = 0; j < buildPoints.Length - 1 - i; j++)
                {
                    if (buildPoints[j].x < buildPoints[j + 1].x)        //만약 i의 거리가 i+1보다 멀다면 i를 i+1로 바꿔야함
                    {
                        Point temp = buildPoints[j];
                        buildPoints[j] = buildPoints[j + 1];
                        buildPoints[j + 1] = temp;
                        continue;
                    }
                }
            }

            if (buildPoints[0].y > buildPoints[1].y)        //만약 i의 거리가 i+1보다 멀다면 i를 i+1로 바꿔야함                
            {
                Point temp = buildPoints[0];
                buildPoints[0] = buildPoints[1];
                buildPoints[1] = temp;
            }
            if (buildPoints[2].y < buildPoints[3].y)        //만약 i의 거리가 i+1보다 멀다면 i를 i+1로 바꿔야함                
            {
                Point temp = buildPoints[2];
                buildPoints[2] = buildPoints[3];
                buildPoints[3] = temp;
            }

            return buildPoints;
        }

        //영역 안에 grass obj 배치하기 grassNum : 설치할 잔디 갯수
        private void assignGrassInGround(IList<RVO.Vector2> vertex,Point center, int grassNum,Transform parentObj)
        {
            for (int i = 0; i < grassNum; i++)
            {
                //x 안에서 선택
                float diffX = Mathf.Abs(vertex[0].x_ - vertex[2].x_);
                float x = UnityEngine.Random.Range(-diffX,diffX)*7;
                //y 안에서 선택
                float diffY = Mathf.Abs(vertex[0].y_ - vertex[2].y_);
                float y = UnityEngine.Random.Range(-diffY, diffY) *7;

                GameObject temp = Instantiate(grassObj[0], new Vector3((float)center.x, (float)center.y, 0), Quaternion.Euler(new Vector3(90, 0, 0)));
                temp.transform.parent = parentObj;
                temp.transform.localPosition = new Vector3(- x, - y, 0);
            }
        }

        public void OnColorButtonClick()
        {
            if (colorIdx <= 2)
                colorIdx++;
            else
                colorIdx = 0;
        }

        public void OnMoveValueChange(InputField input)
        {
            moveSpeed = float.Parse(input.text);
        }

        /// <summary>
        /// Computes the pose.
        /// </summary>
        /// <param name="pattern">Pattern.</param>
        /// <param name="camMatrix">Cam matrix.</param>
        /// <param name="distCoeff">Dist coeff.</param>
        public void computePose(MatOfPoint3f points3d, MatOfPoint2f points2d, Mat camMatrix, MatOfDouble distCoeff)
        {
            Mat Rvec = new Mat();
            Mat Tvec = new Mat();
            Mat raux = new Mat();
            Mat taux = new Mat();
            //objectPoints : 객체 좌표 공간에 객체 점 배열
            //imagePoints : 해당 영상 지점 배열
            //camerMatrix : 입력 카메라 고유 매트릭스
            //distCoeffs : 왜곡 계수의 입력 벡터
            //rvec : 출력 회전 벡터(Rodrigues 참조)는 tvec와 함께 모델 좌표계의 포인트를 카메라 좌표계로 가져옴
            //tvec : 출력 변환 벡터
            Calib3d.solvePnP(points3d, points2d, camMatrix, distCoeff, raux, taux);

            raux.convertTo(Rvec, CvType.CV_32F);
            taux.convertTo(Tvec, CvType.CV_32F);

            Mat rotMat = new Mat(3, 3, CvType.CV_64FC1);
            Calib3d.Rodrigues(Rvec, rotMat);

            pose3d.SetRow(0, new Vector4((float)rotMat.get(0, 0)[0], (float)rotMat.get(0, 1)[0], (float)rotMat.get(0, 2)[0], (float)Tvec.get(0, 0)[0]));
            pose3d.SetRow(1, new Vector4((float)rotMat.get(1, 0)[0], (float)rotMat.get(1, 1)[0], (float)rotMat.get(1, 2)[0], (float)Tvec.get(1, 0)[0]));
            pose3d.SetRow(2, new Vector4((float)rotMat.get(2, 0)[0], (float)rotMat.get(2, 1)[0], (float)rotMat.get(2, 2)[0], (float)Tvec.get(2, 0)[0]));
            pose3d.SetRow(3, new Vector4(0, 0, 0, 1));

            Rvec.Dispose();
            Tvec.Dispose();
            raux.Dispose();
            taux.Dispose();
            rotMat.Dispose();
        }

        public void computeBuildPose(MatOfPoint3f points3d, MatOfPoint2f points2d, Mat camMatrix, MatOfDouble distCoeff,int idx)
        {
            Mat Rvec = new Mat();
            Mat Tvec = new Mat();
            Mat raux = new Mat();
            Mat taux = new Mat();
            //objectPoints : 객체 좌표 공간에 객체 점 배열
            //imagePoints : 해당 영상 지점 배열
            //camerMatrix : 입력 카메라 고유 매트릭스
            //distCoeffs : 왜곡 계수의 입력 벡터
            //rvec : 출력 회전 벡터(Rodrigues 참조)는 tvec와 함께 모델 좌표계의 포인트를 카메라 좌표계로 가져옴
            //tvec : 출력 변환 벡터
            Calib3d.solvePnP(points3d, points2d, camMatrix, distCoeff, raux, taux);

            raux.convertTo(Rvec, CvType.CV_32F);
            taux.convertTo(Tvec, CvType.CV_32F);

            Mat rotMat = new Mat(3, 3, CvType.CV_64FC1);
            Calib3d.Rodrigues(Rvec, rotMat);

            if (idx == 0)       //building
            {
                pose3d = new Matrix4x4();

                pose3d.SetRow(0, new Vector4((float)rotMat.get(0, 0)[0], (float)rotMat.get(0, 1)[0], (float)rotMat.get(0, 2)[0], (float)Tvec.get(0, 0)[0]));
                pose3d.SetRow(1, new Vector4((float)rotMat.get(1, 0)[0], (float)rotMat.get(1, 1)[0], (float)rotMat.get(1, 2)[0], (float)Tvec.get(1, 0)[0]));
                pose3d.SetRow(2, new Vector4((float)rotMat.get(2, 0)[0], (float)rotMat.get(2, 1)[0], (float)rotMat.get(2, 2)[0], (float)Tvec.get(2, 0)[0]));
                pose3d.SetRow(3, new Vector4(0, 0, 0, 1));
            }
            else if (idx == 1)      //plane
            {
                pPose3d = new Matrix4x4();

                pPose3d.SetRow(0, new Vector4((float)rotMat.get(0, 0)[0], (float)rotMat.get(0, 1)[0], (float)rotMat.get(0, 2)[0], (float)Tvec.get(0, 0)[0]));
                pPose3d.SetRow(1, new Vector4((float)rotMat.get(1, 0)[0], (float)rotMat.get(1, 1)[0], (float)rotMat.get(1, 2)[0], (float)Tvec.get(1, 0)[0]));
                pPose3d.SetRow(2, new Vector4((float)rotMat.get(2, 0)[0], (float)rotMat.get(2, 1)[0], (float)rotMat.get(2, 2)[0], (float)Tvec.get(2, 0)[0]));
                pPose3d.SetRow(3, new Vector4(0, 0, 0, 1));
            }
            else if(idx==2)     //tree
            {
                treePose3d = new Matrix4x4();

                treePose3d.SetRow(0, new Vector4((float)rotMat.get(0, 0)[0], (float)rotMat.get(0, 1)[0], (float)rotMat.get(0, 2)[0], (float)Tvec.get(0, 0)[0]));
                treePose3d.SetRow(1, new Vector4((float)rotMat.get(1, 0)[0], (float)rotMat.get(1, 1)[0], (float)rotMat.get(1, 2)[0], (float)Tvec.get(1, 0)[0]));
                treePose3d.SetRow(2, new Vector4((float)rotMat.get(2, 0)[0], (float)rotMat.get(2, 1)[0], (float)rotMat.get(2, 2)[0], (float)Tvec.get(2, 0)[0]));
                treePose3d.SetRow(3, new Vector4(0, 0, 0, 1));
            }

            Rvec.Dispose();
            Tvec.Dispose();
            raux.Dispose();
            taux.Dispose();
            rotMat.Dispose();
        }

        private void UpdateObstalceTransform()
        {
            for (int i = 0; i < buildingObj.Count; i++)
            {
                changeVerticesOfCube(buildingObj[i].transform.GetChild(0).GetComponent<MeshFilter>(), new MatOfPoint2f(buildContours[i].toArray()));
            }
            for (int i = 0; i < treeObj.Count; i++)
            {
                changeVerticesOfCube(treeObj[i].transform.GetChild(0).GetComponent<MeshFilter>(), new MatOfPoint2f(treeContours[i].toArray()));
            }
            for (int i = 0; i < roadObj.Count; i++) 
            {
                changeVerticesOfCube(roadObj[i].transform.GetChild(0).GetComponent<MeshFilter>(), new MatOfPoint2f(roadContours[i].toArray()));
                roadObj[i].transform.GetChild(1).localPosition = -new Vector3((float)roadContoursCenter[i].x, (float)roadContoursCenter[i].y, 0);
            }
        }

        // 큐브의 네꼭지점 변경
        public void changeVerticesOfCube(MeshFilter cubeMesh,MatOfPoint2f src,int isBuilding=0)
        {
            double[] dVert = src.get(0, 0);
            double[] dVert2 = src.get(1, 0);
            double[] dVert3 = src.get(2, 0);
            double[] dVert4 = src.get(3, 0);

            Vector3[] vertice = cubeMesh.mesh.vertices;

            vertice[0] = -new Vector3((float)dVert2[0], (float)dVert2[1], 0.1f);       //1
            vertice[1] = -new Vector3((float)dVert[0], (float)dVert[1], 0.1f);          //0
            vertice[2] = -new Vector3((float)dVert3[0], (float)dVert3[1], 0.1f);        //3
            vertice[3] = -new Vector3((float)dVert4[0], (float)dVert4[1], 0.1f);        //2

            cubeMesh.mesh.vertices = vertice;

        }

        public void changeQuadToCube(GameObject plane,float height)
        {
            //cube object 크기대로 맞추기
            cubeM = plane.transform.GetChild(0).GetComponent<MeshFilter>().mesh;

            c = new Vector3[8];

            vertice = cubeM.vertices;

            c[0] = vertice[0] + Vector3.forward * height;
            c[1] = vertice[1] + Vector3.forward * height;
            c[2] = vertice[3] + Vector3.forward * height;
            c[3] = vertice[2] + Vector3.forward * height;
            c[4] = vertice[0];
            c[5] = vertice[1];
            c[6] = vertice[3];
            c[7] = vertice[2];
 
            vertices = new Vector3[]
            {
                c[0], c[1], c[2], c[3], // Bottom
                c[7], c[4], c[0], c[3], // Left
                c[4], c[5], c[1], c[0], // Front
                c[6], c[7], c[3], c[2], // Back
                c[5], c[6], c[2], c[1], // Right
                c[7], c[6], c[5], c[4]  // Top
            };

            cubeM.Clear();
            cubeM.vertices = vertices;
            cubeM.triangles = triangles;
            cubeM.uv = uvs;
            cubeM.Optimize();
            cubeM.RecalculateNormals();

        }

        //모두 지우기
        public void allDestroy()
        {
            for(int i = treeObj.Count ; i > 0; i--)
            {
                Destroy(treeObj[i - 1]);
            }
            for (int i = buildingObj.Count; i > 0; i--)
            {
                Destroy(buildingObj[i - 1]);
            }
            for (int i = roadObj.Count; i > 0; i--)
            {
                Destroy(roadObj[i - 1]);
            }
            Destroy(crossWalkObj);
            Destroy(planeObj);
            crossWalkObj = null;
            treeObj.Clear();
            treeContours.Clear();
            treeContoursCenter.Clear();
            buildingObj.Clear();
            buildContours.Clear();
            buildContoursCenter.Clear();
            roadObj.Clear();
            roadContours.Clear();
            roadContoursCenter.Clear();
            srcContours.Clear();
            srcContoursCenter.Clear();
        }
         
        //방해물 설정 : 버튼 클릭 함수
        public void clickObstacleMappingButton()
        {
            obstExitUp = null;
            obstExitDown = null;

            //맵핑의 끝을 알려줌
            isObstMapping = true;

            //obstacle들의 material 변경
            for(int i = 0; i < building3DObj.Count; i++)
            {
                building3DObj[i].transform.GetChild(0).GetComponent<MeshRenderer>().material = Transparency;
            }

            //building
            for (int i = 0; i < buildContours.Count; i++)
            {
                double[] dVert = buildContours[i].get(0, 0);
                double[] dVert2 = buildContours[i].get(1, 0);
                double[] dVert3 = buildContours[i].get(2, 0);
                double[] dVert4 = buildContours[i].get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;

                crowdManager.addObstacle(vertex);

                //건물 instance 추가(지붕도 추가)
                GameObject tempObj = Instantiate(ARGameObjectObst, new Vector3(0, 0, 0), Quaternion.Euler(new Vector3(0, 0, 0)));
                tempObj.gameObject.name = "Building" + i;       //생성한 Obj에 이름 붙이기
                GameObject tempRoof = Instantiate(roofPrefab, new Vector3(0, 0, 0), Quaternion.Euler(new Vector3(90, 0, 0)));
                tempRoof.gameObject.name = "Roof" + i;       //생성한 Obj에 이름 붙이기
                tempRoof.transform.parent = tempObj.transform;
                tempRoof.transform.localScale = new Vector3(8, 8, 8);
                tempRoof.transform.localPosition = new Vector3(0, 0, 100);

               
                building3DObj.Add(tempObj);         //빌딩 obj instance 제작

                dVert[1] -= buildContoursCenter[i].y;
                dVert[0] -=  buildContoursCenter[i].x;

                dVert2[1] -=  buildContoursCenter[i].y;
                dVert2[0] -=buildContoursCenter[i].x;

                dVert3[1] -=  buildContoursCenter[i].y;
                dVert3[0] -= buildContoursCenter[i].x;

                dVert4[1] -=  buildContoursCenter[i].y;
                dVert4[0] -=   buildContoursCenter[i].x;

                //cube object 크기대로 맞추기
                cubeM = building3DObj[i].transform.GetChild(0).GetComponent<MeshFilter>().mesh;

                c = new Vector3[8];

                vertice = cubeM.vertices;

                vertice[1] = -new Vector3((float)dVert[0], (float)dVert[1], 0.1f);            //0
                vertice[0] = -new Vector3((float)dVert2[0], (float)dVert2[1], 0.1f);        //1
                vertice[3] = -new Vector3((float)dVert4[0], (float)dVert4[1], 0.1f);        //2
                vertice[2] = -new Vector3((float)dVert3[0], (float)dVert3[1], 0.1f);        //3

                c[0] = new Vector3(vertice[0].x, vertice[0].y, 100.0f);     //1
                c[1] = new Vector3(vertice[1].x, vertice[1].y, 100.0f);     //0
                c[2] = new Vector3(vertice[3].x, vertice[3].y, 100.0f);     //3
                c[3] = new Vector3(vertice[2].x, vertice[2].y, 100.0f);     //2
                c[4] = vertice[0];
                c[5] = vertice[1];
                c[6] = vertice[3];
                c[7] = vertice[2];

                vertices = new Vector3[]
                {
                        c[0], c[1], c[2], c[3], // Bottom
                    c[7], c[4], c[0], c[3], // Left
                    c[4], c[5], c[1], c[0], // Front
                    c[6], c[7], c[3], c[2], // Back
                    c[5], c[6], c[2], c[1], // Right
                    c[7], c[6], c[5], c[4]  // Top
                };

                cubeM.Clear();
                cubeM.vertices = vertices;
                cubeM.triangles = triangles;
                cubeM.Optimize();
                cubeM.RecalculateNormals();

            }
            //tree
            for (int i = 0; i < treeContours.Count; i++)
            {
                double[] dVert = treeContours[i].get(0, 0);
                double[] dVert2 = treeContours[i].get(1, 0);
                double[] dVert3 = treeContours[i].get(2, 0);
                double[] dVert4 = treeContours[i].get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;

                crowdManager.addObstacle(vertex);

                tree3DObj.Add(Instantiate(ARGameObjectTree, new Vector3(0, 0, 0), Quaternion.Euler(new Vector3(0, 0, 0))));

                assignGrassInGround(vertex,treeContoursCenter[i], 20, tree3DObj[tree3DObj.Count - 1].transform);

                dVert[1] -= treeContoursCenter[i].y;
                dVert[0] -= treeContoursCenter[i].x;

                dVert2[1] -= treeContoursCenter[i].y;
                dVert2[0] -= treeContoursCenter[i].x;

                dVert3[1] -= treeContoursCenter[i].y;
                dVert3[0] -= treeContoursCenter[i].x;

                dVert4[1] -= treeContoursCenter[i].y;
                dVert4[0] -= treeContoursCenter[i].x;

                //cube object 크기대로 맞추기
                cubeM = tree3DObj[tree3DObj.Count - 1].transform.GetChild(0).GetComponent<MeshFilter>().mesh;

                c = new Vector3[8];

                vertice = cubeM.vertices;

                vertice[0] = -new Vector3((float)dVert2[0], (float)dVert2[1], 0.1f);       //1
                vertice[1] = -new Vector3((float)dVert[0], (float)dVert[1], 0.1f);          //0
                vertice[2] = -new Vector3((float)dVert3[0], (float)dVert3[1], 0.1f);        //3
                vertice[3] = -new Vector3((float)dVert4[0], (float)dVert4[1], 0.1f);        //2

                c[5] = vertice[1];
                c[4] = vertice[0];
                c[7] = vertice[2];
                c[6] = vertice[3];

                c[1] = new Vector3(vertice[1].x, vertice[1].y, 10.0f);
                c[0] = new Vector3(vertice[0].x, vertice[0].y, 10.0f);
                c[3] = new Vector3(vertice[2].x, vertice[2].y, 10.0f);
                c[2] = new Vector3(vertice[3].x, vertice[3].y, 10.0f);

                vertices = new Vector3[]
                {
                    c[0], c[1], c[2], c[3], // Bottom
                    c[7], c[4], c[0], c[3], // Left
                    c[4], c[5], c[1], c[0], // Front
                    c[6], c[7], c[3], c[2], // Back
                    c[5], c[6], c[2], c[1], // Right
                    c[7], c[6], c[5], c[4]  // Top
                };

                cubeM.Clear();
                cubeM.vertices = vertices;
                cubeM.triangles = triangles;
                cubeM.Optimize();
                cubeM.RecalculateNormals();

            }
            //road
            for (int i = 0; i < roadContours.Count; i++)
            {
                double[] dVert = roadContours[i].get(0, 0);
                double[] dVert2 = roadContours[i].get(1, 0);
                double[] dVert3 = roadContours[i].get(2, 0);
                double[] dVert4 = roadContours[i].get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;

                crowdManager.addObstacle(vertex);
            }

            {
                double[] dVert = planeContour.get(0, 0);
                double[] dVert2 = planeContour.get(1, 0);
                double[] dVert3 = planeContour.get(2, 0);
                double[] dVert4 = planeContour.get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;

                plane3DObj = Instantiate(ARGameObjectPlane, new Vector3(0, 0, 0), Quaternion.Euler(new Vector3(0, 0, 0)));
                plane3DObj.gameObject.name = "Plane";

                dVert[1] -= planeContourCenter.y;
                dVert[0] -= planeContourCenter.x;

                dVert2[1] -= planeContourCenter.y;
                dVert2[0] -= planeContourCenter.x;

                dVert3[1] -= planeContourCenter.y;
                dVert3[0] -= planeContourCenter.x;

                dVert4[1] -= planeContourCenter.y;
                dVert4[0] -= planeContourCenter.x;

                //cube object 크기대로 맞추기
                cubeM = plane3DObj.transform.GetChild(0).GetComponent<MeshFilter>().mesh;

                c = new Vector3[8];

                vertice = cubeM.vertices;

                vertice[0] = -new Vector3((float)dVert2[0], (float)dVert2[1], 0.1f);       //1
                vertice[1] = -new Vector3((float)dVert[0], (float)dVert[1], 0.1f);          //0
                vertice[2] = -new Vector3((float)dVert3[0], (float)dVert3[1], 0.1f);        //3
                vertice[3] = -new Vector3((float)dVert4[0], (float)dVert4[1], 0.1f);        //2

                c[5] = vertice[1];
                c[4] = vertice[0];
                c[7] = vertice[2];
                c[6] = vertice[3];

                c[1] = new Vector3(vertice[1].x, vertice[1].y, 10.0f);
                c[0] = new Vector3(vertice[0].x, vertice[0].y, 10.0f);
                c[3] = new Vector3(vertice[2].x, vertice[2].y, 10.0f);
                c[2] = new Vector3(vertice[3].x, vertice[3].y, 10.0f);

                vertices = new Vector3[]
                {
                    c[0], c[1], c[2], c[3], // Bottom
                    c[7], c[4], c[0], c[3], // Left
                    c[4], c[5], c[1], c[0], // Front
                    c[6], c[7], c[3], c[2], // Back
                    c[5], c[6], c[2], c[1], // Right
                    c[7], c[6], c[5], c[4]  // Top
                };

                cubeM.Clear();
                cubeM.vertices = vertices;
                cubeM.triangles = triangles;
                cubeM.Optimize();
                cubeM.RecalculateNormals();
            }

            srcContours.Clear();
            srcContoursCenter.Clear();

            settingInitialMap();
        }

        public void findExitCirclesButton()
        {
            exitCircles = new Mat();
            {
                Imgproc.HoughCircles(grayMat, exitCircles, Imgproc.CV_HOUGH_GRADIENT, 2, 20, 160, 50, 10, 20);
                pt = new Point();

                obstExitUp = new List<Point>();
                obstExitDown = new List<Point>();
            }
            if (exitCircles != null)
            {
                for (int j = 0; j < exitCircles.cols(); j++)
                {
                    double[] data = exitCircles.get(0, j);
                    pt.x = data[0];
                    pt.y = data[1];

                    if (pt.x != 0 && pt.y != 0)
                    {
                        double rho = data[2];
                        double[] cirColor = hsvMat.get((int)pt.y + (int)rho, (int)pt.x); //원의 위의 점 체크

                        if ((cirColor[0] >= 170 && cirColor[0] <= 180))     //빨강
                        {
                            //지금 현재 exit 갯수
                            obstExitDown.Add(new Point(pt.x, pt.y));
                        }
                        else if ((cirColor[0] >= 60 && cirColor[0] < 100))      //초록
                        {
                            obstExitUp.Add(new Point(pt.x, pt.y));
                        }
                    }
                }

                //만약 2개쌍 이상이면 삭제 한 후 가까운 순서대로 정리
                if (obstExitUp.Count > 0)
                {
                    //다 찾고 나서 짝이 안맞을 경우 뒤에서부터 삭제
                    if (obstExitUp.Count != obstExitDown.Count)
                    {
                        int diff=Mathf.Abs(obstExitDown.Count-obstExitUp.Count);
                        if (obstExitUp.Count > obstExitDown.Count)
                        {
                            for (int i = 0; i < diff; i++)
                            {
                                obstExitUp.RemoveAt(obstExitUp.Count - 1);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < diff; i++)
                            {
                                obstExitDown.RemoveAt(obstExitDown.Count - 1);
                            }
                        }
                    }

                    for (int i = 0; i < obstExitUp.Count-1; i++)
                    {
                        //정렬
                        if (obstExitUp[i].x - obstExitDown[i].x > obstExitUp[i].x - obstExitDown[i + 1].x)
                        {
                            //만약 멀면
                            double temp;
                            temp = obstExitDown[i].x;
                            obstExitDown[i].x = obstExitDown[i + 1].x;
                            obstExitDown[i + 1].x = temp;
                        }
                    }
                }
            }
        }

        // 출구가 있는 건물 맵핑하는 함수
        public void clickExitObstacleMappingButton()
        {
            for (int i = 0; i < obstExitUp.Count; i++)
            {
                //출구랑 가장 가까운 사각형 선별
                for (int j = 0; j < buildContoursCenter.Count; j++)
                {
                    //만약 건물 출구 x - 건물 center x의 차이가 5.0 이하면 해당 출구와 건물은 페어임
                    if (Mathf.Abs((float)(obstExitDown[i].y - buildContoursCenter[j].y)) < 30.0f)
                    {
                        //페어인 경우 idx 저장(재탐색하는 경우도 생각해야함)
                        //자리 바꾸지 말고 idx를 저장하도록 변경
                        idx.Add(i,j);
                    }
                    else
                    {
                        //그 외에는 건물과 출구가 페어가 아님
                    }
                }
            }

            //exit가 있는 건물들 Obstalce delte하기(뒤에서부터 지워야지 idx 겹치지 않음)
            //제일 가까운 obstacle을 찾아내서 지워야함
            crowdManager.deleteObstacles();

            //vertex에 지금 해당하는 obstacle 맵핑하기
            //obstExitDown.count 만큼 생성해야함
            for (int z = 0; z < idx.Count; z++)
            {
                int index = idx[z];

                //vertex에 지금 해당하는 obstacle 위치 넣어주기
                double[] dVert = buildContours[index].get(0, 0);
                double[] dVert2 = buildContours[index].get(1, 0);
                double[] dVert3 = buildContours[index].get(2, 0);
                double[] dVert4 = buildContours[index].get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;

                //만약 obstacle이 이미 입력되있는 상태라면 그 값을 이용하여 작동
                if (vertex.Count == 4)       //obst가 한개인 경우
                {
                    //변수 할당
                    float obstExitUpX = (float)obstExitUp[z].x;
                    float obstExitDownX = (float)obstExitDown[z].x;
                    float obstExitDownY = (float)obstExitDown[z].y;


                    IList<RVO.Vector2> tempVertex = new List<RVO.Vector2>();
                    vertexTop = new List<RVO.Vector2>();
                    vertexBottom = new List<RVO.Vector2>();
                    vertexLeft = new List<RVO.Vector2>();
                    vertexRight = new List<RVO.Vector2>();
                    //4개의 벽 obstacle을 맵핑함

                    for (int j = 0; j < 4; j++)
                        tempVertex.Add(new RVO.Vector2(0, 0));

                    //왼쪽
                    tempVertex[0] = vertex[0];
                    tempVertex[1] = new RVO.Vector2((-obstExitUpX / crowdSize + exitThickness), vertex[0].y_);
                    tempVertex[2] = new RVO.Vector2((-obstExitUpX / crowdSize + exitThickness), vertex[3].y_);
                    tempVertex[3] = vertex[3];

                    for (int j = 0; j < 4; j++)
                        vertexLeft.Add(-tempVertex[j] * crowdSize);

                    crowdManager.addObstacle(tempVertex);
                    //오른쪽
                    tempVertex[0] = vertex[2];
                    tempVertex[1] = vertex[1];
                    tempVertex[2] = vertex[1] - new RVO.Vector2(exitThickness, 0);
                    tempVertex[3] = vertex[2] - new RVO.Vector2(exitThickness, 0);

                    for (int j = 0; j < 4; j++)
                        vertexRight.Add(-tempVertex[j] * crowdSize);

                    crowdManager.addObstacle(tempVertex);

                    //위
                    tempVertex[0] = new RVO.Vector2((-obstExitUpX / crowdSize - exitThickness), vertex[1].y_);
                    tempVertex[1] = vertex[1];
                    tempVertex[2] = vertex[1] - new RVO.Vector2(0, exitThickness);
                    tempVertex[3] = new RVO.Vector2((-obstExitUpX / crowdSize - exitThickness), vertex[1].y_ - exitThickness);

                    for (int j = 0; j < 4; j++)
                        vertexTop.Add(-tempVertex[j] * crowdSize);

                    crowdManager.addObstacle(tempVertex);

                    //아래
                    tempVertex[0] = new RVO.Vector2(vertex[3].x_, (float)(-obstExitDownY / crowdSize - exitThickness));
                    tempVertex[1] = new RVO.Vector2((float)(-obstExitDownX / crowdSize + exitThickness), (-obstExitDownY / crowdSize - exitThickness));
                    tempVertex[2] = tempVertex[1] - new RVO.Vector2(0, exitThickness);
                    tempVertex[3] = vertex[3];

                    for (int j = 0; j < 4; j++)
                        vertexBottom.Add(-tempVertex[j] * crowdSize);

                    crowdManager.addObstacle(tempVertex);
                }
            }

            //나머지 오브젝트에 대한 obstacle 재맵핑
            //건물만 나머지 object를 다시 맵핑하면 됨
            {
                for(int i = 0; i < idx.Count; i++)
                {
                    checkExcept[idx[i]] = true;
                }

                for(int i = 0; i < checkExcept.Length; i++)
                {
                    if (!checkExcept[i])
                        exceptBuildNum = i;
                }

                double[] dVert = buildContours[exceptBuildNum].get(0, 0);
                double[] dVert2 = buildContours[exceptBuildNum].get(1, 0);
                double[] dVert3 = buildContours[exceptBuildNum].get(2, 0);
                double[] dVert4 = buildContours[exceptBuildNum].get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;

                crowdManager.addObstacle(vertex);
         }

            for (int i = 0; i < treeContours.Count; i++)
            {
                double[] dVert = treeContours[i].get(0, 0);
                double[] dVert2 = treeContours[i].get(1, 0);
                double[] dVert3 = treeContours[i].get(2, 0);
                double[] dVert4 = treeContours[i].get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;

                crowdManager.addObstacle(vertex);
            }

            for (int i = 0; i < roadContours.Count; i++)
            {
                double[] dVert = roadContours[i].get(0, 0);
                double[] dVert2 = roadContours[i].get(1, 0);
                double[] dVert3 = roadContours[i].get(2, 0);
                double[] dVert4 = roadContours[i].get(3, 0);

                vertex = new List<RVO.Vector2>();

                for (int j = 0; j < 4; j++)
                    vertex.Add(new RVO.Vector2(0, 0));

                vertex[1] = -new RVO.Vector2((float)dVert[0], (float)dVert[1]) / crowdSize;
                vertex[2] = -new RVO.Vector2((float)dVert2[0], (float)dVert2[1]) / crowdSize;
                vertex[0] = -new RVO.Vector2((float)dVert4[0], (float)dVert4[1]) / crowdSize;
                vertex[3] = -new RVO.Vector2((float)dVert3[0], (float)dVert3[1]) / crowdSize;

                crowdManager.addObstacle(vertex);
            }

        }

            //출발지점 위치 및 목표지점 세팅하기
        public void clickStartPointSettingButton()
        {
            isStartEndPoint = false;
            startPt = null;
            endPt = null;
            circles = new Mat();
            {
                Imgproc.HoughCircles(grayMat, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 20, 160, 50, 5, 20);
                pt = new Point();
            }
            if (circles != null)
            {
                for (int j = 0; j < circles.cols(); j++)
                {
                    double[] data = circles.get(0, j);
                    pt.x = data[0];
                    pt.y = data[1];

                    double rho = data[2];
                    double[] cirColor = hsvMat.get((int)pt.y + (int)rho, (int)pt.x);
                    double[] cirColor2= hsvMat.get((int)pt.y - (int)rho, (int)pt.x);
                    double[] cirColor3 = hsvMat.get((int)pt.y , (int)pt.x + (int)rho);
                    double[] cirColor4 = hsvMat.get((int)pt.y, (int)pt.x - (int)rho);
                    cirColor[0] += cirColor2[0]+cirColor3[0]+cirColor4[0];
                    cirColor[1] += cirColor2[1] + cirColor3[1] + cirColor4[1];
                    cirColor[2] += cirColor2[2] + cirColor3[2] + cirColor4[2];

                    //4개의 점의 색상을 더한 다음 나누기
                    for (int i = 0; i < 3; i++)
                        cirColor[i] /= 4;

                    if ((cirColor[0] > blueMin && cirColor[0] <= blueMax))     //파랑색
                    {
                        startPt = new Point();
                        startPt.x = data[0];
                        startPt.y = data[1];
                    }
                    else if ( (cirColor[0]>=pinkMin && cirColor[0] < pinkMax))       //핑크색
                    {
                        endPt = new Point();
                        endPt.x = data[0];
                        endPt.y = data[1];
                    }
                }
            }

            if (startPt!=null)
                crowdManager.setAgentIntiialPoint((-new RVO.Vector2((float)startPt.x, (float)startPt.y))/crowdSize);
            if (endPt != null)
            {
                if (wayPointConvexContours.Count > 0)     //만약 waypoint가 있다면
                {
                    crowdManager.setAgentGoalPoint((-new RVO.Vector2((float)wayPointConvexContours[0].x, (float)wayPointConvexContours[0].y)) / crowdSize);
                }
                else {              //만약 waypoint가 없다면
                        crowdManager.setAgentGoalPoint((-new RVO.Vector2((float)endPt.x, (float)endPt.y)) / crowdSize);
                 }
            }
            isStartEndPoint = true;
        }

        public void clickWayPointSettingButton()
        {
            wayPointConvexContours.Clear() ;
            wayPointContours.Clear();

            foreach (var cnt in tempContours)
            {
                hull = new MatOfInt();
                convexityDegects = new MatOfInt4();

                Imgproc.convexHull(cnt, hull, false);
                Imgproc.convexityDefects(cnt, hull, convexityDegects);      //여러개 중에서 한개만 확인해줌 (확인할 때 tempContours에 있는거 모두 확인해야함)

                cnt_arr = cnt.toArray();
                distX = new Point();

                //중간 점들 모음 벡터에 저장
                for (int i = 0; i < cnt_arr.Length; i++)
                {
                    double[] colorArray = hsvMat.get((int)cnt_arr[i].y, (int)cnt_arr[i].x);
                    //색상이 검정색일 때 저장함
                    if (colorArray[1] < 20 && colorArray[2] < 100)
                    {
                        bool isFar = true;
                        Vector2 distVec = new Vector2((float)cnt_arr[i].x, (float)cnt_arr[i].y);

                        for (int j = 0; j < wayPointConvexContours.Count; j++)
                        {
                            Vector2 wayPVec = new Vector2((float)wayPointConvexContours[j].x, (float)wayPointConvexContours[j].y) - distVec;
                            //현재 점까지의 거리
                            float dist = Mathf.Sqrt(Mathf.Pow(wayPVec.x, 2) + Mathf.Pow(wayPVec.y, 2));
                            if (dist < 20.0f)
                                isFar = false;
                        }
                        if (isFar)
                            wayPointConvexContours.Add(cnt_arr[i]);
                    }
                }

                if (convexityDegects.size().height > 0)
                {
                    for (int i = 0; i <= convexityDegects.size().height; i++)        //현재는 모든 점들을 다 출력함
                    {
                        double[] dConvex = convexityDegects.get(i, 0);

                        if (dConvex != null)
                        {
                            distX = cnt_arr[(int)dConvex[2]];

                            double[] colorArray = hsvMat.get((int)distX.y, (int)distX.x);

                            if (colorArray[1] < 20 && colorArray[2] <70)            //검정색일때
                            {
                                if (convexityDegects.size().height > 2)
                                {
                                    bool isFar = true;
                                    Vector2 distVec = new Vector2((float)distX.x, (float)distX.y);
                                    //저장하기전 거리가 가까운 점인지 체크
                                    for(int j = 0; j < wayPointConvexContours.Count; j++)
                                    {
                                        Vector2 wayPVec = new Vector2((float)wayPointConvexContours[j].x, (float)wayPointConvexContours[j].y)-distVec;
                                        //현재 점까지의 거리
                                        float dist = Mathf.Sqrt(Mathf.Pow(wayPVec.x, 2) + Mathf.Pow(wayPVec.y, 2));
                                        if (dist < 20.0f)
                                            isFar = false;
                                    }
                                    if(isFar)
                                        wayPointConvexContours.Add(distX);

                                }
                                else
                                {
                                    wayPointConvexContours.Add(distX);
                                }
                            }
                        }
                    }
                }
            }

            
        }

        //way point 보간 함수
        public void clickWPInterpolationButton()
        {
            wayPDist.Clear();
            //start pos와 end pos를 불러오기
            if (startPt != null && endPt != null)      //startPt와 endPt가 존재할 때 실행
            {
                isInterpolation = true;
            }

            //start point 기준 sorting - 거리가 가장 가까운 기준대로 sorting
            //wayPoint의 거리 값 계산 및 저장
            for (int i = 0; i < wayPointConvexContours.Count; i++)
            {
                float dist = dotProductSorting(wayPointConvexContours[i]);

                wayPDist.Add(dist);
            }

            //정렬
            for (int i = 0; i < wayPDist.Count; i++) { 
                for (int j = 0; j < wayPDist.Count-1-i; j++)
                {
                    if (wayPDist[j] > wayPDist[j+1])        //만약 i의 거리가 i+1보다 멀다면 i를 i+1로 바꿔야함                
                    {
                        Point temp = wayPointConvexContours[j];
                        wayPointConvexContours[j] = wayPointConvexContours[j+1];
                        wayPointConvexContours[j+1] = temp;

                        float floatTemp = wayPDist[j];
                        wayPDist[j] = wayPDist[j+1];
                        wayPDist[j + 1] = floatTemp;

                        continue;
                    }
                }
            }
           
            //보간


        }

        //sorting 관련 함수 dot product를 이용하여 start 에서 end까지의 벡터를 정렬
        public float dotProductSorting(Point wayPointPt)
        {
            float dotDist = 0;

            //start 에서 end까지의 벡터
            Vector2 b = new Vector2((float)(endPt.x - startPt.x), (float)(endPt.y - startPt.y));
            //start 에서 end까지의 벡터의 길이
            float distb = Mathf.Sqrt(Mathf.Pow(b.x, 2) + Mathf.Pow(b.y, 2));

            //start에서 계산할 곡점까지의 벡터
            Vector2 a = new Vector2((float)(wayPointPt.x - startPt.x), (float)(wayPointPt.y - startPt.y));

            float dista = Mathf.Sqrt(Mathf.Pow(a.x, 2) + Mathf.Pow(a.y, 2));

            //둘 사이의 degree
            float degree = Mathf.Acos(Vector3.Dot(new Vector3(a.x, a.y), new Vector3(b.x, b.y) / (distb * dista)));

            dotDist = dista * Mathf.Cos(degree);

            return Mathf.Abs(dotDist);
        }
    }
}
