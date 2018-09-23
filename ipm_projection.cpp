#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

struct Camera
{
    Camera() {}
    Camera(const std::string &  cali_file)
    {
        load_from_file(cali_file);
    }
    void load_from_file(const std::string &  cali_file)
    {
        cv::FileStorage fs2(cali_file, cv::FileStorage::READ);
        fs2["model_type"] >> model_type;
        fs2["camera_name"] >> camera_name;
        fs2["image_width"] >> img_width;
        fs2["image_height"] >> img_height;
        fs2["mirror_parameters"]["xi"] >> xi;
        dist_coeffs.resize(4);
        fs2["distortion_parameters"]["k1"] >> dist_coeffs[0];
        fs2["distortion_parameters"]["k2"] >> dist_coeffs[1];
        fs2["distortion_parameters"]["p1"] >> dist_coeffs[2];
        fs2["distortion_parameters"]["p2"] >> dist_coeffs[3];

        K = Eigen::Matrix3f::Identity();
        fs2["projection_parameters"]["gamma1"] >> K(0, 0);
        fs2["projection_parameters"]["gamma2"] >> K(1, 1);
        fs2["projection_parameters"]["u0"] >> K(0, 2);
        fs2["projection_parameters"]["v0"] >> K(1, 2);
    }

    void distortion(float mx_u, float my_u,
                                float *dx_u, float *dy_u) {
        float mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

        mx2_u = mx_u * mx_u;
        my2_u = my_u * my_u;
        mxy_u = mx_u * my_u;
        rho2_u = mx2_u + my2_u;
        rad_dist_u = dist_coeffs[0] * rho2_u + dist_coeffs[1] * rho2_u*rho2_u;
        *dx_u = mx_u * rad_dist_u + 2 * dist_coeffs[2] *mxy_u + dist_coeffs[3] * (rho2_u + 2 * mx2_u);
        *dy_u = my_u * rad_dist_u + 2 * dist_coeffs[3] *mxy_u + dist_coeffs[2] * (rho2_u + 2 * my2_u);
    }

    Eigen::Vector3f projectPoint(const Eigen::Vector3f & p)
    {
        Eigen::Vector3f ret;
        // Project points to the normalised plane
        float norm = p.norm();
        float z = p.z() + xi * norm;

        
        float mx_u = p.x() / z;
        float my_u = p.y() / z;

        {
            // Apply distortion
            float dx_u, dy_u;
            distortion(mx_u, my_u, &dx_u, &dy_u);
            ret.x() = mx_u + dx_u;
            ret.y() = my_u + dy_u;
            ret.z() = 1.0f;
        }

        // Apply generalised projection matrix
        // Matlab points start at 1
        return K * ret;
    }

    int32_t img_width;
    int32_t img_height;
    std::string camera_name;
    std::string model_type;
    Eigen::Matrix3f K;
    float xi;
    std::vector<float> dist_coeffs; /// k1 k2 p1 p2
    Eigen::Matrix4f T_glob_cam = Eigen::Matrix4f::Identity();
};

Eigen::Matrix3f calculate_T_H_pCuR(
    const Eigen::Matrix4f T_glob_cam,
    const Eigen::Matrix3f & K)
{


    Eigen::Matrix4f T_uCuW = T_glob_cam.inverse();

    Eigen::Matrix<float, 4, 3> T_uR4uR3;
    T_uR4uR3 <<
        1, 0, 0,
        0, 1, 0,
        0, 0, 0,
        0, 0, 1;

    Eigen::Matrix<float, 3, 4> T_uW3uW4;
    T_uW3uW4 <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0;


    //rotation around z
    Eigen::Matrix3f T_perm;
    T_perm <<
        0, 1, 0,
        1, 0, 0,
        0, 0, 1;


    Eigen::Matrix3f T_H_pCuR = K * T_uW3uW4 * T_uCuW * T_uR4uR3 * T_perm;
    return T_H_pCuR;
}


Eigen::Matrix3f calculate_T_mCuR(
    const Eigen::Matrix4f T_glob_cam)
{
    Eigen::Matrix4f T_uCuW = T_glob_cam.inverse();

    Eigen::Matrix<float, 4, 3> T_uR4uR3;
    T_uR4uR3 <<
        1, 0, 0,
        0, 1, 0,
        0, 0, -1,
        0, 0, 1;

    //rotation around z
    Eigen::Matrix3f T_perm;
    T_perm <<
        0, 1, 0,
        1, 0, 0,
        0, 0, 1;

    Eigen::Matrix<float, 3, 4> T_uW3uW4;
    T_uW3uW4 <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0;

    Eigen::Matrix3f T_mCuR = T_uW3uW4 * T_uCuW * T_uR4uR3 * T_perm;
    return T_mCuR;
}


Eigen::Matrix3f calculate_T_H_pCpR(
    const Eigen::Matrix4f T_glob_cam,
    const Eigen::Matrix3f & K,
    const Eigen::Matrix3f & T_uRpR)
{
    return calculate_T_H_pCuR(T_glob_cam, K) * T_uRpR;
}

Camera * getCamByName(const std::string & name, std::vector<Camera> & cameras)
{
    for (auto & c : cameras)
        if (c.camera_name == name)
            return &c;
    return nullptr;
}
int main(int argc, const char * argv[])
{

    std::vector<std::string> calib_f_names = 
    {
        "e:/tmp/2018-01-24_03-33-33/output/F_FISHEYE_C_calib.yaml",
        "e:/tmp/2018-01-24_03-33-33/output/M_FISHEYE_R_calib.yaml",
        "e:/tmp/2018-01-24_03-33-33/output/B_FISHEYE_C_calib.yaml",
        "e:/tmp/2018-01-24_03-33-33/output/M_FISHEYE_L_calib.yaml"
    };
    //std::vector<std::string> calib_f_names =
    //{
    //    "e:/tmp/2018-01-24_03-33-33/calib/camera_0_calib.yaml",
    //    "e:/tmp/2018-01-24_03-33-33/calib/camera_1_calib.yaml",
    //    "e:/tmp/2018-01-24_03-33-33/calib/camera_2_calib.yaml",
    //    "e:/tmp/2018-01-24_03-33-33/calib/camera_3_calib.yaml"
    //};
    std::string extrinsics_calib = "e:/tmp/2018-01-24_03-33-33/output/extrinsic.txt";
    //std::string extrinsics_calib = "i:/2018-01-24_03-33-33/output/extrinsic.txt";
    
    std::vector<Camera> cameras(calib_f_names.size());

    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].load_from_file(calib_f_names[i]);


    std::ifstream ext_f(extrinsics_calib);
    std::string line;
    while (!ext_f.eof())
    {
        std::getline(ext_f, line);
        auto c = getCamByName(line, cameras);
        if (c != nullptr)
        {
            for (int i = 0; i < 3; ++i)
            {
                std::getline(ext_f, line);
                std::stringstream ss(line);
                ss >> c->T_glob_cam(i, 0) >> c->T_glob_cam(i, 1) >> c->T_glob_cam(i, 2) >> c->T_glob_cam(i, 3);
            }
            std::getline(ext_f, line);
        }
        else
        {
            std::getline(ext_f, line);
            std::getline(ext_f, line);
            std::getline(ext_f, line);
            std::getline(ext_f, line);
        }
    }

    std::vector<std::string> video_names = 
    {
        "e:/tmp/2018-01-24_03-32-28/stream05Rec-F_FISHEYE_C_nvidia.-2018-01-24_03-32-28.h264",
        "e:/tmp/2018-01-24_03-32-28/stream03Rec-M_FISHEYE_R_nvidia.-2018-01-24_03-32-28.h264",
        "e:/tmp/2018-01-24_03-32-28/stream04Rec-B_FISHEYE_C_nvidia.-2018-01-24_03-32-28.h264",
        "e:/tmp/2018-01-24_03-32-28/stream02Rec-M_FISHEYE_L_nvidia.-2018-01-24_03-32-28.h264"
    };

    std::vector<cv::VideoCapture> captures(video_names.size());
    for (size_t i = 0; i < captures.size(); ++i)
    {
        captures[i].open(video_names[i]);
    }

    cv::Mat f, r, b, l;
    cv::Mat fs, rs, bs, ls;
    cv::Mat ipm(900,1280, CV_8UC3);
    float px_to_meter = 1/40.0f;
    Eigen::Matrix3f T_pRuR;
    T_pRuR << 
        1 / px_to_meter, 0, 640.0f, 
        0, 1 / px_to_meter, 450.0f, 
        0, 0, 1.0f;
    Eigen::Matrix3f T_uRpR = T_pRuR.inverse();

   

    while (captures[0].read(f) && captures[1].read(r) && captures[2].read(b) && captures[3].read(l))
    {
        ipm.setTo(0);
        if (f.rows == cameras[0].img_height && f.cols == cameras[0].img_width)
        {
            fs = f;
            rs = r;
            bs = b;
            ls = l;
        }
        else
        {
            cv::resize(f, fs, cv::Size(cameras[0].img_width, cameras[0].img_height));
            cv::resize(r, rs, cv::Size(cameras[0].img_width, cameras[0].img_height));
            cv::resize(b, bs, cv::Size(cameras[0].img_width, cameras[0].img_height));
            cv::resize(l, ls, cv::Size(cameras[0].img_width, cameras[0].img_height));
        }

//        Eigen::Matrix3f R= Eigen::AngleAxisf(3.141592f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
//        Eigen::Matrix4f R4 = Eigen::Matrix4f::Identity();
//        R4.block<3, 3>(0, 0) = R;
        //for (size_t i = 0; i < cameras.size(); ++i)
        for (size_t i = 0; i < 4; ++i)
        {
            Eigen::Matrix3f T_mCpR = calculate_T_mCuR(cameras[i].T_glob_cam) * T_uRpR;
            cv::Mat img;
            switch (i)
            {
            case 0:
                img = fs;
                break;
            case 1:
                img = rs;
                break;
            case 2:
                img = bs;
                break;
            case 3:
                img = ls;
                break;
            default:
                assert(false);
                break;
            }
            for (int y = 0; y < ipm.rows; ++y)
            {
                for (int x = 0; x < ipm.cols; ++x)
                {
                    Eigen::Vector3f p3d = T_mCpR * Eigen::Vector3f(x, y, 1.0f);
                    if (p3d.z() > 0 && p3d.x() > - 4 *std::abs(p3d.z()) && p3d.x() <  4 * std::abs(p3d.z()))
                    {
                        Eigen::Vector3f p = cameras[i].projectPoint(p3d);
                        if (p.x() < fs.cols && p.x() > 0 && p.y() < fs.rows && p.y() > 0)
                        {
                            cv::Vec3b c = img.at<cv::Vec3b>(p.y(), p.x());
                            auto curr = ipm.at<cv::Vec3b>(y, x);
                            if (curr[0] == 0)
                                ipm.at<cv::Vec3b>(y, x) = c;
                            else
                                ipm.at<cv::Vec3b>(y, x) = c/2 + curr/2;

                        }
                    }
                }
            }

        }

        cv::imshow("ipm", ipm);
        int key = cv::waitKey(1);
        if (key == 27)
            break;

    }

    return 0;
}
