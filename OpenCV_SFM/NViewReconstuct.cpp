#include <iostream>
#include <vector>
#include <io.h>
#include <fstream>
#include <string>
#include <queue>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "ceres/ceres.h" 
#include "ceres/rotation.h"
#include "glog/logging.h" 


using namespace std;
using namespace cv;


struct Pt3DPly
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float nx = 0.0f;
	float ny = 0.0f;
	float nz = 0.0f;
	uint8_t r = 0;
	uint8_t g = 0;
	uint8_t b = 0;
};

// --------------------
int getAllFiles(const string& path, const string& format, vector<string>& files);


int init_structure(
	const Mat& K,
	const vector<vector<KeyPoint>>& key_points_for_all,
	const vector<vector<Vec3b>>& colors_for_all,
	const vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
);
void get_matched_points(
	const vector<KeyPoint>& p1,
	const vector<KeyPoint>& p2,
	const vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2);
void get_matched_colors(
	const vector<Vec3b>& c1,
	const vector<Vec3b>& c2,
	const vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
);
bool find_transform(const Mat& K,
	const vector<Point2f>& p1,
	const vector<Point2f>& p2,
	Mat& R,
	Mat& T,
	Mat& mask);

void maskout_points(const Mat& mask, vector<Point2f>& p1);
void maskout_2d_pts_pair(const Mat& mask, vector<Point2f>& pts1, vector<Point2f>& pts2);

void maskout_colors(const Mat& mask, vector<Vec3b>& p1);

void bundle_adjustment(
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure
);

int reconstruct(const Mat& K,
	Mat& R1, Mat& T1, Mat& R2, Mat& T2,
	vector<Point2f>& p1, vector<Point2f>& p2,
	vector<Point3d>& structure);
void get_obj_pts_and_img_pts(
	const vector<DMatch>& matches,
	const vector<int>& struct_indices,
	const vector<Point3d>& structure,
	const vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points);
void fuse_structure(
	const vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors);
void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
);
void match_features(const Mat& query, const Mat& train, vector<DMatch>& matches);
void match_features_for_all(const vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all);
void save_structure(string file_name,
	vector<Mat>& rotations,
	vector<Mat>& motions,
	vector<Point3d>& structure,
	vector<Vec3b>& colors);

void write_ply_binary(const std::string& path,
	const std::vector<Pt3DPly>& points);
int get_ply_pts3d(const vector<Point3d>& pts3d,
	const vector<Point3d>& normals,
	const vector<Vec3b>& colors,
	vector<Pt3DPly>& pts3d_ply);

// ��ȡbundler.out�ļ�(SFM������ļ�), ���ply����...
int read_bundler_write_ply(const std::string& ply_out_f_path,
	const std::string& img_dir_path,
	const std::string& bundler_out_path,
	const std::string& bunlder_list_path);

int PCAFitPlane(const vector<Point3d>& pts3d, double* normal);
int PCAFitPlane(const vector<Pt3DPly>& pts3d, double* normal);

// ���Ʒ���������
int estimate_normals(const vector<Point3d>& pts3d,
	const int K,
	vector<Point3d>& normals);

int estimate_normals(const vector<Pt3DPly>& pts3d,
	const int K,
	vector<Point3d>& normals);

// --------------------

struct ReprojectCost
{
	cv::Point2d m_observation;

	ReprojectCost(cv::Point2d& observation)
		: m_observation(observation)
	{
	}

	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pt3d, T* residuals) const
	{
		const T* r = extrinsic;      // ��ת����ָ��
		const T* t = &extrinsic[3];  // ƽ������ָ��

		// Apply rotation: from world to camera
		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pt3d, pos_proj);

		// Apply the camera translation
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		// Convert to normalized coordinates in camera coordinate system
		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];

		// Apply intrinsic: convert to pixel coordinate system
		const T u = fx * x + cx;
		const T v = fy * y + cy;

		residuals[0] = u - T(m_observation.x);
		residuals[1] = v - T(m_observation.y);

		return true;
	}
};

void save_structure(string file_name,
	vector<Mat>& rotations,
	vector<Mat>& motions,
	vector<Point3d>& structure,
	vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; i++)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}

void write_ply_binary(const std::string& path,
	const std::vector<Pt3DPly>& points)
{
	std::fstream text_file(path, std::ios::out);
	assert(text_file.is_open());

	size_t valid_cnt = 0;
	for (const auto& point : points)
	{
		if (isnan(point.x) || isnan(point.y)
			|| isnan(point.z) || isnan(point.nx)
			|| isnan(point.ny) || isnan(point.nz))
		{
			//std::cout << "[Nan]: " << point.x << " " << point.y << " "
			//	<< point.z << " " << point.nx << " " << point.ny << " "
			//	<< point.nz << std::endl;
			continue;
		}

		valid_cnt += 1;
	}

	text_file << "ply" << std::endl;
	text_file << "format binary_little_endian 1.0" << std::endl;
	//text_file << "element vertex " << points.size() << std::endl;
	text_file << "element vertex " << valid_cnt << std::endl;
	text_file << "property float x" << std::endl;
	text_file << "property float y" << std::endl;
	text_file << "property float z" << std::endl;
	text_file << "property float nx" << std::endl;
	text_file << "property float ny" << std::endl;
	text_file << "property float nz" << std::endl;
	text_file << "property uchar red" << std::endl;
	text_file << "property uchar green" << std::endl;
	text_file << "property uchar blue" << std::endl;
	text_file << "end_header" << std::endl;
	text_file.close();

	std::fstream binary_file(path,
		std::ios::out | std::ios::binary | std::ios::app);
	assert(binary_file.is_open());

	for (const auto& point : points)
	{
		if (isnan(point.x) || isnan(point.y) || isnan(point.z)
			|| isnan(point.nx) || isnan(point.ny) || isnan(point.nz))
		{
			//std::cout << "[Nan]: " << point.x << " " << point.y << " "
			//	<< point.z << " " << point.nx << " " << point.ny
			//	<< point.nz << std::endl;
			continue;
		}

		binary_file.write((char *)(&(point.x)), sizeof(float));
		binary_file.write((char *)(&(point.y)), sizeof(float));
		binary_file.write((char *)(&(point.z)), sizeof(float));
		binary_file.write((char *)(&(point.nx)), sizeof(float));
		binary_file.write((char *)(&(point.ny)), sizeof(float));
		binary_file.write((char *)(&(point.nz)), sizeof(float));
		binary_file.write((char *)(&(point.r)), sizeof(uint8_t));
		binary_file.write((char *)(&(point.g)), sizeof(uint8_t));
		binary_file.write((char *)(&(point.b)), sizeof(uint8_t));
	}

	binary_file.close();
}

int get_ply_pts3d(const vector<Point3d>& pts3d,
	const vector<Point3d>& normals,
	const vector<Vec3b>& colors,
	vector<Pt3DPly>& pts3d_ply)
{
	if (pts3d.size() != normals.size()
		|| pts3d.size() != colors.size())
	{
		printf("[Err]: items size not equal.\n");
		return -1;
	}

	// �����ڴ�
	pts3d_ply.resize(pts3d.size());

	for (int i = 0; i < pts3d.size(); ++i)
	{
		const Point3d& pt3d = pts3d[i];
		const Point3d& normal = normals[i];
		const Vec3b& color = colors[i];

		// д��������
		Pt3DPly pt3dply;

		pt3dply.x = (float)pt3d.x;
		pt3dply.y = (float)pt3d.y;
		pt3dply.z = (float)pt3d.z;


		pt3dply.nx = (float)normal.x;
		pt3dply.ny = (float)normal.y;
		pt3dply.nz = (float)normal.z;

		pt3dply.b = color[0];
		pt3dply.g = color[1];
		pt3dply.r = color[2];

		pts3d_ply[i] = pt3dply;
	}
	printf("Total %zd 3D points.\n", pts3d.size());

	return 0;
}

int read_bundler_write_ply(const std::string& ply_out_f_path,
	const std::string& img_dir_path,
	const std::string& bundler_file_path,
	const std::string& bunlder_list_path)
{
	std::ifstream f_bundler(bundler_file_path);
	std::ifstream f_list(bunlder_list_path);

	if (!f_bundler.is_open())
	{
		printf("[Err]: bundler_file_path wrong.\n");
		exit(-1);
	}
	if (!f_list.is_open())
	{
		printf("[Err]: bundler_list_path wrong.\n");
		exit(-1);
	}

	// Read Header line
	std::string header;
	std::getline(f_bundler, header);

	// Read image number and points number
	int num_images, num_points;
	f_bundler >> num_images >> num_points;

	// Read each image(view) info
	for (int img_id = 0; img_id < num_images; ++img_id)
	{
		// Read image name(path)
		std::string img_path;
		std::getline(f_list, img_path);
		img_path = img_dir_path + "/" + img_path;

		// Read image
		cv::Mat img = cv::imread(img_path);

		// ---------- Read camera matrix
		float K[9] = {
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 1.0f
		};
		float f_len;
		f_bundler >> f_len;
		if (f_len < 1.0f)
		{
			K[0] = f_len * float(img.cols);
			K[4] = f_len * float(img.rows);
		}
		else
		{
			f_bundler >> K[0];
			K[4] = K[0];  // fx == fy
		}

		printf("Focus length(pixel): %.3f\n", K[0]);

		// Read image center
		K[2] = img.cols / 2.0f;  // cx
		K[5] = img.rows / 2.0f;  // cy

		// Read radial distortion coefficients(k1, k2)
		float k[2];
		f_bundler >> k[0] >> k[1];

		// ----------

		// ---------- Read poses
		// ----- Read rotation matrix
		float R[9];
		for (size_t i = 0; i < 9; ++i)
		{
			f_bundler >> R[i];
		}
		for (size_t i = 3; i < 9; ++i)  // why?
		{
			R[i] = -R[i];
		}

		// ----- Read translation vector
		float T[3];
		f_bundler >> T[0] >> T[1] >> T[2];
		T[1] = -T[1];
		T[2] = -T[2];
	}

	// load points info
	std::vector<Pt3DPly> pts3d_ply(num_points);
	for (int pt_id = 0; pt_id < num_points; ++pt_id)
	{
		// 3d point coordinates
		Pt3DPly pt3d;
		f_bundler >> pt3d.x >> pt3d.y >> pt3d.z;

		// 3d point color
		f_bundler >> pt3d.r >> pt3d.g >> pt3d.b;

		// track length
		int track_len;
		f_bundler >> track_len;

		// load 3D point track info
		std::vector<int> track;
		track.resize(track_len);
		for (int i = 0; i < track_len; ++i)
		{
			int feature_idx;
			float im_x, im_y;

			f_bundler >> track[i] >> feature_idx >> im_x >> im_y;
		}

		// Add pt3d to pts vector
		pts3d_ply[pt_id] = pt3d;
	}

	// estimate point normals
	std::vector<cv::Point3d> normals;
	estimate_normals(pts3d_ply, 10, normals);

	// write out point cloud(.ply)
	write_ply_binary(ply_out_f_path, pts3d_ply);

	return 0;
}

struct Pt3dDist
{
	bool operator ()(const pair<Point3d, Point3d> dist_0, const pair<Point3d, Point3d>& dist_1)
	{
		const double d0 = sqrt((dist_0.first.x - dist_0.second.x) * (dist_0.first.x - dist_0.second.x)
			+ (dist_0.first.y - dist_0.second.y) * (dist_0.first.y - dist_0.second.y)
			+ (dist_0.first.z - dist_0.second.z) * (dist_0.first.z - dist_0.second.z));

		const double d1 = sqrt((dist_1.first.x - dist_1.second.x) * (dist_1.first.x - dist_1.second.x)
			+ (dist_1.first.y - dist_1.second.y) * (dist_1.first.y - dist_1.second.y)
			+ (dist_1.first.z - dist_1.second.z) * (dist_1.first.z - dist_1.second.z));

		return d0 > d1;
	}
};

struct Pt3dPlyDist
{
	bool operator ()(const pair<Pt3DPly, Pt3DPly> dist_0, const pair<Pt3DPly, Pt3DPly>& dist_1)
	{
		const double d0 = sqrt((dist_0.first.x - dist_0.second.x) * (dist_0.first.x - dist_0.second.x)
			+ (dist_0.first.y - dist_0.second.y) * (dist_0.first.y - dist_0.second.y)
			+ (dist_0.first.z - dist_0.second.z) * (dist_0.first.z - dist_0.second.z));

		const double d1 = sqrt((dist_1.first.x - dist_1.second.x) * (dist_1.first.x - dist_1.second.x)
			+ (dist_1.first.y - dist_1.second.y) * (dist_1.first.y - dist_1.second.y)
			+ (dist_1.first.z - dist_1.second.z) * (dist_1.first.z - dist_1.second.z));

		return d0 > d1;
	}
};

int estimate_normals(const vector<Pt3DPly>& pts3d,
	const int K,
	vector<Point3d>& normals)
{
	for (int i = 0; i < pts3d.size(); ++i)
	{
		// ��ǰ����3D��
		const Pt3DPly& pt3d = pts3d[i];

		priority_queue<pair<Pt3DPly, Pt3DPly>,
			vector<pair<Pt3DPly, Pt3DPly>>, Pt3dPlyDist> neighbors;

		// ----- �����(K)����
		for (int j = 0; j < pts3d.size(); ++j)
		{
			if (j != i)
			{
				neighbors.push(make_pair(pt3d, pts3d[j]));
			}
		}

		// ȡ��������С��top K
		std::vector<Pt3DPly> neighs(K);
		for (int k = 0; k < K; ++k)
		{
			const pair<Pt3DPly, Pt3DPly>& neighbor = neighbors.top();
			neighs[k] = neighbor.second;  // ȡ�����Ž�neighs����

			//// ��֤����...
			//double dist = sqrt((neighs[k].x - pt3d.x) * (neighs[k].x - pt3d.x)
			//+ (neighs[k].y - pt3d.y) * (neighs[k].y - pt3d.y)
			//+ (neighs[k].z - pt3d.z) * (neighs[k].z - pt3d.z));
			//printf("Dist : %.3f\n", dist);

			neighbors.pop();
		}
		//printf("Nearest %d neighbor of 3D point %d computed.\n", K, i);

		// ƽ�����
		double normal[3] = { 0.0 };
		PCAFitPlane(neighs, normal);

		normals[i].x = normal[0];
		normals[i].y = normal[1];
		normals[i].z = normal[2];
	}

	return 0;
}

// ���Ʒ��������� 
int estimate_normals(const vector<Point3d>& pts3d,
	const int K,  // top K
	vector<Point3d>& normals)
{
	for (int i = 0; i < pts3d.size(); ++i)
	{
		// ��ǰ����3D��
		const Point3d& pt3d = pts3d[i];

		priority_queue<pair<Point3d, Point3d>, vector<pair<Point3d, Point3d>>, Pt3dDist> neighbors;

		// ----- �����(K)����
		for (int j = 0; j < pts3d.size(); ++j)
		{
			if (j != i)
			{
				neighbors.push(make_pair(pt3d, pts3d[j]));
			}
		}

		// ȡ��������С��top K
		vector<Point3d> neighs(K);
		for (int k = 0; k < K; ++k)
		{
			const pair<Point3d, Point3d>& neighbor = neighbors.top();
			neighs[k] = neighbor.second;  // ȡ�����Ž�neighs����

			//// ��֤����...
			//double dist = sqrt((neighs[k].x - pt3d.x) * (neighs[k].x - pt3d.x)
			//+ (neighs[k].y - pt3d.y) * (neighs[k].y - pt3d.y)
			//+ (neighs[k].z - pt3d.z) * (neighs[k].z - pt3d.z));
			//printf("Dist : %.3f\n", dist);

			neighbors.pop();
		}
		//printf("Nearest %d neighbor of 3D point %d computed.\n", K, i);

		// ƽ�����
		double normal[3] = { 0.0 };
		PCAFitPlane(neighs, normal);

		normals[i].x = normal[0];
		normals[i].y = normal[1];
		normals[i].z = normal[2];
	}

	return 0;
}

int PCAFitPlane(const vector<Point3d>& pts3d, double* normal)
{
	double ave_x = 0.0f, ave_y = 0.0f, ave_z = 0.0f;
	for (auto pt : pts3d)
	{
		ave_x += pt.x;
		ave_y += pt.y;
		ave_z += pt.z;
	}
	ave_x /= double(pts3d.size());
	ave_y /= double(pts3d.size());
	ave_z /= double(pts3d.size());

	// ��Э�������A
	Eigen::Matrix3d A;
	double sum_xx = 0.0f, sum_yy = 0.0f, sum_zz = 0.0f,
		sum_xy = 0.0f, sum_xz = 0.0f, sum_yz = 0.0f;
	for (auto pt : pts3d)
	{
		sum_xx += (pt.x - ave_x) * (pt.x - ave_x);
		sum_yy += (pt.y - ave_y) * (pt.y - ave_y);
		sum_zz += (pt.z - ave_z) * (pt.z - ave_z);

		sum_xy += (pt.x - ave_x) * (pt.y - ave_y);
		sum_xz += (pt.x - ave_x) * (pt.z - ave_z);
		sum_yz += (pt.y - ave_y) * (pt.z - ave_z);
	}
	A(0, 0) = sum_xx / double(pts3d.size());  // ��ʵ, û��Ҫ���ֵ
	A(0, 1) = sum_xy / double(pts3d.size());
	A(0, 2) = sum_xz / double(pts3d.size());
	A(1, 0) = sum_xy / double(pts3d.size());
	A(1, 1) = sum_yy / double(pts3d.size());
	A(1, 2) = sum_yz / double(pts3d.size());
	A(2, 0) = sum_xz / double(pts3d.size());
	A(2, 1) = sum_yz / double(pts3d.size());
	A(2, 2) = sum_zz / double(pts3d.size());

	// ��Э�������A������ֵ����������
	Eigen::EigenSolver<Eigen::Matrix3d> ES(A);
	Eigen::MatrixXcd eigen_vals = ES.eigenvalues();
	Eigen::MatrixXcd eigen_vects = ES.eigenvectors();
	Eigen::MatrixXd eis = eigen_vals.real();
	Eigen::MatrixXd vects = eigen_vects.real();

	// ����С����ֵ��Ӧ����������
	Eigen::MatrixXf::Index min_idx, max_idx;
	eis.rowwise().sum().minCoeff(&min_idx);
	eis.rowwise().sum().maxCoeff(&max_idx);

	// ----- ������ֵ(��������)���򣺴�С����
	int mid_idx = 0;
	if (0 == (int)min_idx)
	{
		mid_idx = max_idx == 1 ? 2 : 1;
	}
	else if (1 == (int)min_idx)
	{
		mid_idx = max_idx == 0 ? 2 : 0;
	}
	else
	{
		mid_idx = max_idx == 0 ? 1 : 0;
	}

	// ��С����ֵ���õ���������
	double& a = vects(0, min_idx);
	double& b = vects(1, min_idx);
	double& c = vects(2, min_idx);

	// ȷ����ȷ�ķ�����: ȷ��normalָ��camera
	if (a * ave_x + b * ave_y + c * ave_z > 0.0f)
	{
		a = -a;
		b = -b;
		c = -c;
	}

	// ��С��������(������)L2��һ��
	const double DENOM = sqrt(a*a + b * b + c * c);
	a /= DENOM;
	b /= DENOM;
	c /= DENOM;
	double plane_normal[3] = { a, b, c };

	// ����ƽ�淨����
	memcpy(normal, plane_normal, sizeof(double) * 3);

	return 0;
}

int PCAFitPlane(const vector<Pt3DPly>& pts3d, double * normal)
{
	double ave_x = 0.0f, ave_y = 0.0f, ave_z = 0.0f;
	for (auto pt : pts3d)
	{
		ave_x += pt.x;
		ave_y += pt.y;
		ave_z += pt.z;
	}
	ave_x /= double(pts3d.size());
	ave_y /= double(pts3d.size());
	ave_z /= double(pts3d.size());

	// ��Э�������A
	Eigen::Matrix3d A;
	double sum_xx = 0.0f, sum_yy = 0.0f, sum_zz = 0.0f,
		sum_xy = 0.0f, sum_xz = 0.0f, sum_yz = 0.0f;
	for (auto pt : pts3d)
	{
		sum_xx += (pt.x - ave_x) * (pt.x - ave_x);
		sum_yy += (pt.y - ave_y) * (pt.y - ave_y);
		sum_zz += (pt.z - ave_z) * (pt.z - ave_z);

		sum_xy += (pt.x - ave_x) * (pt.y - ave_y);
		sum_xz += (pt.x - ave_x) * (pt.z - ave_z);
		sum_yz += (pt.y - ave_y) * (pt.z - ave_z);
	}
	A(0, 0) = sum_xx / double(pts3d.size());  // ��ʵ, û��Ҫ���ֵ
	A(0, 1) = sum_xy / double(pts3d.size());
	A(0, 2) = sum_xz / double(pts3d.size());
	A(1, 0) = sum_xy / double(pts3d.size());
	A(1, 1) = sum_yy / double(pts3d.size());
	A(1, 2) = sum_yz / double(pts3d.size());
	A(2, 0) = sum_xz / double(pts3d.size());
	A(2, 1) = sum_yz / double(pts3d.size());
	A(2, 2) = sum_zz / double(pts3d.size());

	// ��Э�������A������ֵ����������
	Eigen::EigenSolver<Eigen::Matrix3d> ES(A);
	Eigen::MatrixXcd eigen_vals = ES.eigenvalues();
	Eigen::MatrixXcd eigen_vects = ES.eigenvectors();
	Eigen::MatrixXd eis = eigen_vals.real();
	Eigen::MatrixXd vects = eigen_vects.real();

	// ����С����ֵ��Ӧ����������
	Eigen::MatrixXf::Index min_idx, max_idx;
	eis.rowwise().sum().minCoeff(&min_idx);
	eis.rowwise().sum().maxCoeff(&max_idx);

	// ----- ������ֵ(��������)���򣺴�С����
	int mid_idx = 0;
	if (0 == (int)min_idx)
	{
		mid_idx = max_idx == 1 ? 2 : 1;
	}
	else if (1 == (int)min_idx)
	{
		mid_idx = max_idx == 0 ? 2 : 0;
	}
	else
	{
		mid_idx = max_idx == 0 ? 1 : 0;
	}

	// ��С����ֵ���õ���������
	double& a = vects(0, min_idx);
	double& b = vects(1, min_idx);
	double& c = vects(2, min_idx);

	// ȷ����ȷ�ķ�����: ȷ��normalָ��camera
	if (a * ave_x + b * ave_y + c * ave_z > 0.0f)
	{
		a = -a;
		b = -b;
		c = -c;
	}

	// ��С��������(������)L2��һ��
	const double DENOM = sqrt(a*a + b * b + c * c);
	a /= DENOM;
	b /= DENOM;
	c /= DENOM;
	double plane_normal[3] = { a, b, c };

	// ����ƽ�淨����
	memcpy(normal, plane_normal, sizeof(double) * 3);

	return 0;
}

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat img;

	// ��ȡͼ�񣬻�ȡͼ�������㲢����
	Ptr<Feature2D> feature_extractor = cv::SIFT::create();  // cv::AKAZE::create();
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		// ��ȡͼ��
		img = imread(*it);
		if (img.empty())
		{
			continue;
		}
		cout << "Extracting features for image " << *it << "..." << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;

		// ż�������ڴ����ʧ�ܵĴ���  Detects keypoints and computes the descriptors
		// feature_extractor->detectAndCompute(image, noArray(), key_points, descriptor);
		feature_extractor->detect(img, key_points);
		feature_extractor->compute(img, key_points, descriptor);

		// ��������٣����ų���ͼ��
		if (key_points.size() <= 10)
		{
			continue;
		}
		else
		{
			printf("%zd 2D feature point detected.\n", key_points.size());
		}

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		// ��ͨ�� ��Ÿ�λ����ͨ����ɫ
		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& kp = key_points[i].pt;
			const int& y = int(kp.y);
			const int& x = int(kp.x);
			if (y <= img.rows && x <= img.cols)
			{
				colors[i] = img.at<Vec3b>(y, x);
			}
			else
			{
				printf("[Warning]: pt2d[%.3f, %.3f] out of image range.\n", kp.x, kp.y);
			}
		}

		colors_for_all.push_back(colors);
	}
}

void match_features_for_all(const vector<Mat>& descriptor_for_all,
	vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();

	// n��ͼ������˳���� n-1 ��ƥ��
	// 1��2ƥ�䣬2��3ƥ�䣬3��4ƥ�䣬�Դ�����
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;

		vector<DMatch> matches;
		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);

		if (matches.size() == 0)
		{
			printf("[Warning]: zero matches between %d and %d.\n", i, i + 1);
		}

		matches_for_all.push_back(matches);
	}
}

void match_features(const Mat& query, const Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);  // ȡtop2ƥ����õ�ƥ��

	// ��ȡ����Ratio Test����Сƥ��ľ���
	float min_dist = FLT_MAX;
	for (int i = 0; i < knn_matches.size(); ++i)
	{
		// Rotio Test
		if (knn_matches[i][0].distance > 0.6 * knn_matches[i][1].distance)
		{
			continue;
		}

		float dist = knn_matches[i][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	matches.clear();
	for (size_t i = 0; i < knn_matches.size(); ++i)
	{
		// �ų�������Ratio Test�ĵ��ƥ��������ĵ�
		if (
			knn_matches[i][0].distance > 0.6 * knn_matches[i][1].distance ||
			knn_matches[i][0].distance > 5 * max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// ����ƥ���
		matches.push_back(knn_matches[i][0]);
	}
}

// ��ǰ��֡ͼ�������ά�ؽ�
int init_structure(
	const Mat& K,
	const vector<vector<KeyPoint>>& key_points_for_all,  // ÿһ֡��ȡ��������
	const vector<vector<Vec3b>>& colors_for_all,
	const vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
)
{
	// ����ǰ��֡(���0��1)ͼ��֮��ı任����
	vector<Point2f> pts2d_1, pts2d_2;
	vector<Vec3b> c2;
	Mat R, T;	// ��ת�����ƽ������
	Mat mask;	// mask�д�����ĵ����ƥ��㣬������ĵ����ʧ���
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], pts2d_1, pts2d_2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);

	// ����ƥ�����������㱾�ʾ��󲢷ֽ�õ�R,T����
	find_transform(K, pts2d_1, pts2d_2, R, T, mask);
	//printf("mask type: %d\n", mask.type());  // mask: CV_8U(uint8)

	// ---------- �ų�ƥ�䲻��ȷ�ĵ�: ����mask
	//// maskout_points(mask, p1);
	//// maskout_points(mask, p2);
	maskout_2d_pts_pair(mask, pts2d_1, pts2d_2);
	maskout_colors(mask, colors);

	// ǰ��֡���ǲ����õ�ǰ��֡��3D��: ������Ч��2D����ƥ���
	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	const int ret = reconstruct(K, R0, T0, R, T, pts2d_1, pts2d_2, structure);
	if (ret < 0)
	{
		return ret;
	}

	// ����任����
	rotations = { R0, R };  // ��ʼ��
	motions = { T0, T };

	// ��correspond_struct_idx�Ĵ�С��ʼ��Ϊ��key_points_for_all��ȫһ��
	correspond_struct_idx.clear();  // ͨ��2D keypoint����3D�������
	correspond_struct_idx.resize(key_points_for_all.size());  // N frames
	for (size_t fr_i = 0; fr_i < key_points_for_all.size(); ++fr_i)
	{
		// ��ʼ��Ϊ-1
		correspond_struct_idx[fr_i].resize(key_points_for_all[fr_i].size(), -1);
	}

	// ��дǰ��֡(frame 0 and frame 1)�Ľṹ����: 2D���������� ����> 3D������
	const vector<DMatch>& matches = matches_for_all[0];  //total (N-1) matches for N frames

	int idx = 0;
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
		{
			continue;
		}

		// ����������Ӧ��idx���, ����������ͬһ������, idx����structure�ж�Ӧ�Ŀռ����������
		correspond_struct_idx[0][matches[i].queryIdx] = idx;
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}

	printf("Total %d 3D points from the first two frames' valid keypoint matches.\n", idx);
	return 0;
}

void get_matched_points(
	const vector<KeyPoint>& p1,
	const vector<KeyPoint>& p2,
	const vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	const vector<Vec3b>& c1,
	const vector<Vec3b>& c2,
	const vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

bool find_transform(const Mat& K,
	const vector<Point2f>& p1, const vector<Point2f>& p2,
	Mat& R, Mat& T,
	Mat& mask)
{
	// �����ڲ��������ȡ����Ľ���͹�������(��������)
	const double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	const Point2d principle_point(K.at<double>(2), K.at<double>(5));

	// ����ƥ�����ȡ��������ʹ��RANSAC����һ���ų�ʧ���: �ڲ���ͬ
	const Mat& E = cv::findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty())
	{
		return false;
	}

	double feasible_count = countNonZero(mask);	// �õ�����Ԫ�أ��������е���Ч��
	// cout << (int)feasible_count << " - in - " << p1.size() << endl;

	// ����RANSAC���ԣ�outlier��������50%ʱ������ǲ��ɿ���
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
	{
		return false;
	}

	// �ֽⱾ�����󣬻�ȡ��Ա任: �ڲ���ͬ
	int pass_count = cv::recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	// cout << "pass_count = " << pass_count << endl;

	// ͬʱλ���������ǰ���ĵ������Ҫ�㹻��
	if (((double)pass_count) / feasible_count < 0.7)
	{
		return false;
	}

	return true;
}

void maskout_points(const Mat& mask, vector<Point2f>& p1)
{
	vector<Point2f> p1_copy = p1;  // �˴���û��ʹ����ֵ����&, �ʲ���ָ��ͬһ���ڴ��ַ
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void maskout_2d_pts_pair(const Mat& mask, vector<Point2f>& pts1, vector<Point2f>& pts2)
{
	vector<Point2f> pts1_copy = pts1;  // ������ֵ(��ֵ)
	vector<Point2f> pts2_copy = pts2;

	pts1.clear();
	pts2.clear();

	// Ԥ�ȷ�����ڴ棬����push_back�׶ζ�η����ڴ�
	pts1.reserve(pts1_copy.size());
	pts2.reserve(pts2_copy.size());

	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.at<uchar>(i) > 0)
		{
			pts1.push_back(pts1_copy[i]);  // ����mask: size++
			pts2.push_back(pts2_copy[i]);
		}
		else
		{
			printf("2D point id %d is masked out.\n", i);
		}
	}
}

void maskout_colors(const Mat& mask, vector<Vec3b>& pts_color)
{
	vector<Vec3b> p1_copy = pts_color;
	pts_color.clear();
	pts_color.reserve(p1_copy.size());

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			pts_color.push_back(p1_copy[i]);
		}
	}
}

int reconstruct(const Mat& K,
	Mat& R1, Mat& T1, Mat& R2, Mat& T2,
	vector<Point2f>& pts2d_1, vector<Point2f>& pts2d_2,
	vector<Point3d>& structure)
{
	if (pts2d_1.size() == 0 || pts2d_2.size() == 0)
	{
		printf("[Err]: empty 2d points.\n");
		return -1;
	}

	// ���������ͶӰ����[R, T], triangulatePointsֻ֧��float/double��
	Mat proj_1(3, 4, CV_32FC1);
	Mat proj_2(3, 4, CV_32FC1);

	R1.convertTo(proj_1(Range(0, 3), Range(0, 3)), CV_32FC1);  // Range: [start, end)
	//T1.convertTo(proj_2(Range(0, 3), Range(3, 4)), CV_32FC1);
	T1.convertTo(proj_1.col(3), CV_32FC1);

	R2.convertTo(proj_2(Range(0, 3), Range(0, 3)), CV_32FC1);
	//T2.convertTo(proj_2(Range(0, 3), Range(3, 4)), CV_32FC1);
	T2.convertTo(proj_2.col(3), CV_32FC1);  // [R|T]

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj_1 = fK * proj_1;
	proj_2 = fK * proj_2;

	// ���ǲ����ؽ�3D����
	cv::Mat pts4d;  // 4��N:ÿ��һ��3D���������(4ά)
	cv::triangulatePoints(proj_1, proj_2, pts2d_1, pts2d_2, pts4d);

	structure.clear();
	structure.reserve(pts4d.cols);  // Ԥ�ȷ�����ڴ�, �����ڴ��η���
	for (int i = 0; i < pts4d.cols; ++i)
	{
		const Mat_<float>& pt4d_homo = pts4d.col(i);  // ÿһ�д���һ����
		pt4d_homo /= pt4d_homo(3);	// ������ꡪ����>���������
		structure.push_back(Point3f(pt4d_homo(0), pt4d_homo(1), pt4d_homo(2)));
	}

	return 0;
}

void bundle_adjustment(
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure
)
{
	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}

	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

	// loss function make bundle adjustment robuster.
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);

	// load points
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
	{
		vector<int>& point3d_ids = correspond_struct_idx[img_idx];
		vector<KeyPoint>& key_points = key_points_for_all[img_idx];
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
		{
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
			{
				continue;
			}

			Point2d observed = key_points[point_idx].pt;

			// ģ������У���һ��Ϊ���ۺ��������ͣ��ڶ���Ϊ���۵�ά�ȣ�ʣ�������ֱ�Ϊ���ۺ�����һ�ڶ����е�����������ά��
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),			// Intrinsic
				extrinsics[img_idx].ptr<double>(),	// View Rotation and Translation
				&(structure[point3d_id].x)			// Point in 3D space
			);
		}
	}

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 4;  //�����ĸ��߳�
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (!summary.IsSolutionUsable())
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
	}
}

void get_obj_pts_and_img_pts(
	const vector<DMatch>& matches,  // matches for this frame
	const vector<int>& struct_indices,  // struct indices for keypoints of this frame
	const vector<Point3d>& structure,
	const vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		const int& query_idx = matches[i].queryIdx;
		const int& train_idx = matches[i].trainIdx;

		const int& struct_idx = struct_indices[query_idx];
		if (struct_idx < 0)	// ������ǰһ��ͼ��û��ƥ���
		{
			continue;
		}

		object_points.push_back(structure[struct_idx]);

		// train�ж�Ӧ�ؼ��������(2D)
		image_points.push_back(key_points[train_idx].pt);
	}
}

void fuse_structure(
	const vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		const int& query_idx = matches[i].queryIdx;
		const int& train_idx = matches[i].trainIdx;

		const int& struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)  // ���õ��ڿռ����Ѿ����ڣ������ƥ����Ӧ�Ŀռ��Ӧ����ͬһ��������Ҫ��ͬ
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		// ���õ��ڿռ���δ���ڣ����õ���뵽�ṹ�У������ƥ���Ŀռ�������Ϊ�¼���ĵ������
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = int(structure.size()) - 1;
	}
}

//��ȡ�ض���ʽ���ļ���    
int getAllFiles(const string& path, const string& format, vector<string>& files)
{
	intptr_t hFile = 0;  // �ļ����  64λ��long ��Ϊ intptr_t
	struct _finddata_t fileinfo;  // �ļ���Ϣ 
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1)  // �ļ�����
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))  // �ж��Ƿ�Ϊ�ļ���
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)  // �ļ������в���"."��".."
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));  // �����ļ�����
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), format, files);  // �ݹ�����ļ���
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));  // ��������ļ��У������ļ���
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	return int(files.size());
}

// --------------------

int main(int argc, char** argv)
{
	vector<string> img_names;
	const std::string img_dir = std::string(argv[1]);
	const string format = std::string(".jpg");  // .jpg .png
	const int N_files = getAllFiles(img_dir, format, img_names);
	printf("Total %d image files.\n", N_files);

	// ����ڲμҾ���
	//Mat K(Matx33d(
	//	2759.48, 0, 1520.69,
	//	0, 2764.16, 1006.81,
	//	0, 0, 1));
	Mat K(Matx33d(
		2826.561, 0, 1835.259,
		0, 2826.519, 1370.103,
		0, 0, 1));

	// TODO: ��ζ�ȡͼƬmetadata, ����������ڲξ���K...

	vector<vector<cv::KeyPoint>> kpts_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;

	// ��ȡ����ͼ�������
	extract_features(img_names, kpts_for_all, descriptor_for_all, colors_for_all);

	// ������ͼ�����˳�ε�����ƥ��
	match_features_for_all(descriptor_for_all, matches_for_all);

	vector<Point3d> structure;
	vector<vector<int>> correspond_struct_idx;	// �����i��ͼ���е�j�������Ӧ��structure��3D�������
	vector<Vec3b> colors;  // 3���ֽڱ�ʾһ��RGB��BGR��ɫ
	vector<Mat> rotations;  // ��ת��������
	vector<Mat> translations;  // ƽ����������

	// ��ʼ���ṹ(��ά����)ǰ��֡
	printf("\nConstruct from the first two frames...\n");
	const int ret = init_structure(
		K,
		kpts_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		translations
	);

	// ������ʽ�ؽ�ʣ���ͼ��
	printf("\nIncremental SFM...\n");
	for (int i = 1; i < matches_for_all.size(); ++i)  // ����ʣ��ƥ��
	{
		vector<Point3f> obj_pts;  // 3D��
		vector<Point2f> img_pts;  // 2D��
		Mat r, R, T;

		// ��ȡ��i��ͼ����ƥ����Ӧ����ά�㣬�Լ��ڵ�i+1��ͼ���ж�Ӧ�����ص�
		get_obj_pts_and_img_pts(
			matches_for_all[i],  // ��i��ƥ��: ��i֡�͵�i+1֡��ƥ��
			correspond_struct_idx[i],
			structure,
			kpts_for_all[i + 1],
			obj_pts,
			img_pts
		);

		// ��⵱ǰ֡(��i+1֡)�����λ��(�任����[R|T])
		if (obj_pts.size() < 4 || img_pts.size() < 4)
		{
			printf("[Warning]: too few 3D-2D point pairs for frame %d.\n", i);
			continue;
		}
		const bool& ret = cv::solvePnPRansac(obj_pts, img_pts, K, cv::noArray(), r, T);

		// ����ת����ת��Ϊ��ת����
		cv::Rodrigues(r, R);  // CV�ṩ������ת��������ת�����໥ת���ĺ���

		// ���浱ǰ֡�ı任����
		rotations.push_back(R);  // ��i+1֡����ת����
		translations.push_back(T);  // ��i+1֡��ƽ������

		// ��ȡ��i֡�͵�i+1֡��ƥ��������
		vector<Point2f> pts2d_1, pts2d_2;
		vector<Vec3b> colors_1, colors_2;
		get_matched_points(kpts_for_all[i],
			kpts_for_all[i + 1],
			matches_for_all[i],
			pts2d_1, pts2d_2);
		get_matched_colors(colors_for_all[i],
			colors_for_all[i + 1],
			matches_for_all[i],
			colors_1, colors_2);

		// ����֮ǰ��õ�R, T������ά�ؽ�
		vector<Point3d> next_structure;
		reconstruct(K, rotations[i], translations[i], R, T, pts2d_1, pts2d_2, next_structure);
		printf("Frame %d reconstructed.\n", i);

		//���µ��ؽ�3D�������Ѿ��ؽ���3D���ƽ����ں�
		fuse_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i + 1],
			structure,
			next_structure,
			colors,
			colors_1
		);
		printf("Frame %d point cloud fused, total %d points now.\n", i, (int)structure.size());
	}

	// �����Ż�ǰ�Ľ��
	save_structure("../Viewer/structure.yml", rotations, translations, structure, colors);

	// �������(�Ż�)
	printf("\nBundle adjustment fo SFM...\n");
	Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	cout << "intrinsic:\n" << intrinsic << endl;
	vector<Mat> extrinsics;  // �����������
	for (size_t i = 0; i < rotations.size(); ++i)
	{
		Mat extrinsic(6, 1, CV_64FC1);
		Mat r;
		Rodrigues(rotations[i], r);

		r.copyTo(extrinsic.rowRange(0, 3));  // ǰ��������ת����
		translations[i].copyTo(extrinsic.rowRange(3, 6));  // ��������ƽ������

		// ����������
		extrinsics.push_back(extrinsic);
	}

	// do bundle adjustment
	bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, kpts_for_all, structure);

	// ����������
	vector<Point3d> normals(structure.size());
	estimate_normals(structure, 10, normals);

	// �����Ż���Ľ��
	save_structure("../Viewer/structure_ba.yml", rotations, translations, structure, colors);
	printf("structure_ba.yml saved.\n");

	printf("Saving structure to ply...\n");
	vector<Pt3DPly> pts3dply;
	get_ply_pts3d(structure, normals, colors, pts3dply);
	write_ply_binary(string("../Viewer/structure_ba.ply"), pts3dply);
	printf("../Viewer/structure_ba.ply saved.\n");

	cout << "Save structure done." << endl;

	//read_bundler_write_ply("E:/CppProjs/OpencvSFM/dataset/desktop/desktop.ply",
	//	"E:/CppProjs/OpencvSFM/dataset/desktop/",
	//	"E:/CppProjs/OpencvSFM/dataset/desktop/desktop.out",
	//	"E:/CppProjs/OpencvSFM/dataset/desktop/desktop.txt");

	getchar();

	return 0;
}