#include <stdio.h>
#include <random>
#include <chrono>
#include <filesystem>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "TorchNoWarnings.h"
#include "Critic.cuh"
#include <Windows.h>
#include <fstream>
#undef min
#undef max

auto device = torch::kCUDA;

const int SIZE_TENSOR_ACTIONS = 2;
const int SIZE_TENSOR_STATES = 6;
const int FIELD_ACTION_SIZE = 50;
const float DT = 0.01f;
const float MAX_COORD = 250.f;
const float MAX_FORCE = 50.f;
const float MAX_VELOCITY = 50.f;
const float MAX_ANGULAR_VELOCITY = 15.f;

// Hyperparameters
int STEP_COUNT_IN_ACTIONS_FIELD = 5;
//

// Only for GUI
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> disCoords(-MAX_COORD, MAX_COORD);
const float DRONE_START_X = disCoords(gen);
const float DRONE_START_Y = disCoords(gen);
const float DRONE_START_VX = 0.f;
const float DRONE_START_VY = 0.f;
const float DRONE_START_PITCH = 0.f;
const float DRONE_START_ANGULAR_V = 0.f;
const float DRONE_WIDTH = 30.f;
const float DRONE_HEIGHT = 10.f;
int OLD_MOUSE_X, OLD_MOUSE_Y, NEW_MOUSE_X, NEW_MOUSE_Y;
auto actionGUI = torch::tensor({ MAX_FORCE, MAX_FORCE }, torch::device(device)).view({ 1, -1 });
auto droneStateGUI = torch::tensor({ DRONE_START_X, DRONE_START_Y, DRONE_START_VX, DRONE_START_VY, DRONE_START_PITCH, DRONE_START_ANGULAR_V }, torch::device(device)).view({ 1, -1 });
float DRONE_HALF_HEIGHT = DRONE_HEIGHT / 2.f;
auto startTime = std::chrono::high_resolution_clock::now();
float DRONE_HALF_WIDTH = DRONE_WIDTH / 2.f;
float DRONE_HALF_DIAGONAL = pow(DRONE_HALF_WIDTH * DRONE_HALF_WIDTH + DRONE_HALF_HEIGHT * DRONE_HALF_HEIGHT, 0.5f);

std::vector<std::vector<float>> rocket;
std::vector<std::vector<int>> faces;
//


torch::Tensor UpdatePhysics(const torch::Tensor& states, const torch::Tensor& actions)
{
	// initialize constants
	static const float MASS = 1.f;
	static const float C_WINDAGE = 0.5f;
	static const float S_FRONTAL = 0.08f; // 200mm x 400mm
	static const float S_MOTOR = 0.0054f; // (15mm x 30mm (motor S) + 150mm x 15mm (ray S)) * 2
	static const float AIR_DENSITY = 1.29f;
	static const float FORCE_AIR_COEF_LINEAR = C_WINDAGE * S_FRONTAL * AIR_DENSITY / 2.f; // 0.0258
	static const float FORCE_AIR_COEF_ANGULAR = C_WINDAGE * S_MOTOR * AIR_DENSITY / 2.f; // 0.0017415
	static const float GRAVITY_CONSTANT = 9.8f;
	static const float I = MASS * DRONE_HALF_WIDTH * DRONE_HALF_WIDTH / 3.f; // moment of inertia
	// unpacking state
	auto statesSplit = torch::split(states, { 1, 1, 1, 1, 1, 1 }, 1);
	auto x = statesSplit[0]; // x coords
	auto y = statesSplit[1]; // y coords
	auto vX = statesSplit[2]; // velocity by x
	auto vY = statesSplit[3]; // velocity by y
	auto pitch = statesSplit[4]; // tangage angle
	auto velocityAngular = statesSplit[5]; // angular velocity
	// calculate external forces
	auto windageX = FORCE_AIR_COEF_LINEAR * vX * vX;
	auto windageY = FORCE_AIR_COEF_LINEAR * vY * vY;
	auto windageAngular = FORCE_AIR_COEF_ANGULAR * velocityAngular * velocityAngular * DRONE_HALF_WIDTH * DRONE_HALF_WIDTH;
	windageX = torch::where(vX > 0.f, -windageX, windageX);
	windageY = torch::where(vY > 0.f, -windageY, windageY);
	windageAngular = torch::where(velocityAngular > 0.f, -windageAngular, windageAngular);
	auto gravityY = -MASS * GRAVITY_CONSTANT;
	// unpacking action
	auto actionsSplit = torch::split(actions, { 1, 1 }, 1);
	auto thrust1 = actionsSplit[0] * MAX_FORCE; // new thrust motor 1 (left)
	auto thrust2 = actionsSplit[1] * MAX_FORCE; // new thrust motor 2 (right)
	// update angular velocity
	auto accelerationAngular = DRONE_HALF_WIDTH * (thrust1 - thrust2 + windageAngular) / I;
	auto velocityAngularNew = velocityAngular + accelerationAngular * DT;
	// update linear velocity
	auto localThrust = thrust1 + thrust2;
	auto thrustX = torch::sin(pitch) * localThrust; // 90 - angle
	auto thrustY = torch::cos(pitch) * localThrust; // 90 - angle
	auto accelerationX = (thrustX + windageX) / MASS; // linear acceleration by x
	auto accelerationY = (thrustY + windageY + gravityY) / MASS; // linear acceleration by y
	auto vXNew = vX + accelerationX * DT;
	auto vYNew = vY + accelerationY * DT;
	// update orientation (tangage angle)
	auto anglePitchNew = pitch + velocityAngularNew * DT;
	anglePitchNew = torch::where(anglePitchNew > M_PI, anglePitchNew - M_PI * 2.f, anglePitchNew);
	anglePitchNew = torch::where(anglePitchNew < -M_PI, anglePitchNew + M_PI * 2.f, anglePitchNew);
	// update coords
	auto xNew = x + vXNew * DT;
	auto yNew = y + vYNew * DT;

	return torch::cat({ xNew, yNew, vXNew, vYNew, anglePitchNew, velocityAngularNew }, 1);;
}


torch::Tensor Update(const torch::Tensor& states, const torch::Tensor& actions)
{
	auto statesSplit = torch::split(states, { SIZE_TENSOR_STATES, 1 }, 1);
	auto statesClear = statesSplit[0];
	auto statesTTL = statesSplit[1];
	auto nextStates = UpdatePhysics(statesClear, actions);
	return torch::cat({ nextStates, statesTTL - 1.f }, 1);
}


struct CriticNetImpl : torch::nn::Module
{
	torch::Tensor SCALE;

	CriticNetImpl()
		: V1(SIZE_TENSOR_STATES, 256) // {x, y, vX, vY, Theta, w}
		, V2(256, 256)
		, V3(256, 272)

		, AV1(2, 16)
		, AV2(16, 16)
		, AV3(16, 1)

		, Q(256, 1)
	{
		SCALE = torch::tensor({ {1.f / MAX_COORD, 1.f / MAX_COORD, 1.f / MAX_VELOCITY, 1.f / MAX_VELOCITY, 1.f / (float)M_PI, 1.f / MAX_ANGULAR_VELOCITY} }, torch::device(device));
		register_buffer("SCALE", SCALE);

		register_module("V1", V1);
		register_module("V2", V2);
		register_module("V3", V3);
		register_module("AV1", AV1);
		register_module("AV2", AV2);
		register_module("AV3", AV3);
		register_module("Q", Q);
		torch::nn::init::xavier_uniform_(V1->weight, torch::nn::init::calculate_gain(torch::kReLU));
		torch::nn::init::xavier_uniform_(V2->weight, torch::nn::init::calculate_gain(torch::kReLU));
		torch::nn::init::xavier_uniform_(V3->weight, torch::nn::init::calculate_gain(torch::kLinear));
		torch::nn::init::xavier_uniform_(AV1->weight, torch::nn::init::calculate_gain(torch::kReLU));
		torch::nn::init::xavier_uniform_(AV2->weight, torch::nn::init::calculate_gain(torch::kReLU));
		torch::nn::init::xavier_uniform_(AV3->weight, torch::nn::init::calculate_gain(torch::kLinear));
		torch::nn::init::xavier_uniform_(Q->weight, torch::nn::init::calculate_gain(torch::kLinear));
		torch::nn::init::zeros_(V1->bias);
		torch::nn::init::zeros_(V2->bias);
		torch::nn::init::zeros_(V3->bias);
		torch::nn::init::zeros_(AV1->bias);
		torch::nn::init::zeros_(AV2->bias);
		torch::nn::init::zeros_(AV3->bias);
		torch::nn::init::zeros_(Q->bias);
	}


	std::tuple<torch::Tensor, torch::Tensor> ForwardHalf(const torch::Tensor& states)
	{
		auto statesScaled = states * SCALE;

		auto x1 = torch::relu(V1->forward(statesScaled));
		auto x2 = torch::relu(V2->forward(x1));
		auto q = Q->forward(x2);
		auto approx = V3->forward(x2 + x1);

		return std::make_tuple(q, approx);
	}

	torch::Tensor forward(const torch::Tensor& statesActions)
	{
		auto c = statesActions.split({ SIZE_TENSOR_STATES, SIZE_TENSOR_ACTIONS }, 1);
		auto states = c[0];
		auto actions = c[1];

		auto [q, approx] = ForwardHalf(states);
		c = approx.split({ 256, 16 }, 1);
		auto approxWeight = c[0].view({ -1, 16, 16 });
		auto approxBias = c[1].view({ -1, 16, 1 });

		auto x = torch::relu(AV1->forward(actions));
		x = torch::relu((torch::bmm(approxWeight, x.view({ -1, 16, 1 })) + approxBias).squeeze());
		auto qA = AV3->forward(x);

		return q + qA;
	}

	torch::Tensor forwardNoGrad(const torch::Tensor& stateAction)
	{
		torch::NoGradGuard noGrad;
		return forward(stateAction);
	}

	void copyParametersFrom(const auto& networkParameters, const float polyak)
	{
		for (const auto& parameter : networkParameters)
		{
			auto selfValue = named_parameters()[parameter.key()];
			selfValue.set_data(polyak * parameter.value() + (1 - polyak) * selfValue);
		}
	}

	torch::nn::Linear V1;
	torch::nn::Linear V2;
	torch::nn::Linear V3;
	torch::nn::Linear AV1;
	torch::nn::Linear AV2;
	torch::nn::Linear AV3;
	torch::nn::Linear Q;
};
TORCH_MODULE(CriticNet);


torch::Tensor ActionsWithoutActor(const torch::Tensor& states, CriticNet& CriticNetwork)
{
	auto [q, approx] = CriticNetwork->ForwardHalf(states);

	const int64_t BATCH_SIZE = states.size(0);
	auto actionsMax = torch::zeros({ BATCH_SIZE, SIZE_TENSOR_ACTIONS }, states.options()) + 0.5f;
	float kScale = 1.0f;
	for (int step = 0; step < STEP_COUNT_IN_ACTIONS_FIELD; step++, kScale *= 0.25f)
	{
		auto actionsNoise = torch::rand({ BATCH_SIZE * FIELD_ACTION_SIZE, SIZE_TENSOR_ACTIONS }, states.options()) - 0.5f;
		auto actions = actionsNoise.view({ BATCH_SIZE, FIELD_ACTION_SIZE, SIZE_TENSOR_ACTIONS }) * kScale + actionsMax.view({ -1, 1, SIZE_TENSOR_ACTIONS });
		actions = actions.view({ BATCH_SIZE * FIELD_ACTION_SIZE, SIZE_TENSOR_ACTIONS }).clamp(0.0f, 1.0f);

		auto x = torch::relu(CriticNetwork->AV1->forward(actions));
		x = torch::relu(BatchLinearCUDA(approx, 16, 16, x));

		auto qs = CriticNetwork->AV3->forward(x);
		actionsMax = ActionMaxCUDA(qs, actions, FIELD_ACTION_SIZE);
	}

	return actionsMax;
}


void MyInitField()
{
	int windowHalfWidth = glutGet(GLUT_WINDOW_WIDTH) / 2;
	int windowHalfHeight = glutGet(GLUT_WINDOW_HEIGHT) / 2;
	glClearColor(0.7f, 0.7f, 0.7f, 1);
	glOrtho(-windowHalfWidth, windowHalfWidth, -windowHalfHeight, windowHalfHeight, -1, 1);
	glutWarpPointer(windowHalfWidth, windowHalfHeight);
	OLD_MOUSE_X = windowHalfWidth;
	OLD_MOUSE_Y = windowHalfHeight;
	NEW_MOUSE_X = windowHalfWidth;
	NEW_MOUSE_Y = windowHalfHeight;
	
	float k = 30.f;
	std::string line;
	std::ifstream file("rocket.obj");
	while (std::getline(file, line))
	{
		std::stringstream ss;
		std::replace(line.begin(), line.end(), '/', ' ');
		ss << line;
		std::string control;
		ss >> control;
		if (control == "v")
		{
			std::vector<float> vertice;
			float a, b, c;
			ss >> a >> b >> c;
			vertice.emplace_back(a * k);
			vertice.emplace_back(b * k);
			vertice.emplace_back(c * k);
			rocket.emplace_back(vertice);
		}
		if (control == "f")
		{
			std::vector<int> face;
			int a, b, c;
			while (true)
			{
				if (ss.eof()) break;
				ss >> a >> b >> c;
				face.emplace_back(a);
			}
			faces.emplace_back(face);
		}
	}
}


void Draw()
{
	glClear(GL_COLOR_BUFFER_BIT);
	int diffX = NEW_MOUSE_X - OLD_MOUSE_X;
	int diffY = NEW_MOUSE_Y - OLD_MOUSE_Y;
	glTranslatef(diffX, diffY, 0.f);
	OLD_MOUSE_X = NEW_MOUSE_X;
	OLD_MOUSE_Y = NEW_MOUSE_Y;
	float DRONE_START_X = droneStateGUI[0][0].item<float>() - diffX;
	float DRONE_START_Y = droneStateGUI[0][1].item<float>() - diffY;
	auto newCoords = torch::tensor({ { DRONE_START_X, DRONE_START_Y } }, torch::device(device));
	droneStateGUI = torch::cat({ newCoords, droneStateGUI.slice(1, 2) }, 1);

	float JET_FORCE1 = actionGUI[0][0].item<float>();
	float JET_FORCE2 = actionGUI[0][1].item<float>();
	float JET_LEN1 = JET_FORCE1 * 2.f;
	float JET_LEN2 = JET_FORCE2 * 2.f;
	float THETA = droneStateGUI[0][4].item<float>();
	float JET_ANGLE = M_PI / 2.f - THETA;

	float droneRadStep = 0.32175f; // arccos(DRONE_HALF_WIDTH / DRONE_HALF_DIAGONAL)
	// draw drone
	float cosJet = cos(JET_ANGLE);
	float sinJet = sin(JET_ANGLE);
	float droneLeftBottomX = DRONE_START_X + DRONE_HALF_DIAGONAL * cos(JET_ANGLE + M_PI / 2.f + droneRadStep);
	float droneLeftBottomY = DRONE_START_Y + DRONE_HALF_DIAGONAL * sin(JET_ANGLE + M_PI / 2.f + droneRadStep);
	float droneLeftTopX = DRONE_START_X + DRONE_HALF_DIAGONAL * cos(JET_ANGLE + M_PI / 2.f - droneRadStep);
	float droneLeftTopY = DRONE_START_Y + DRONE_HALF_DIAGONAL * sin(JET_ANGLE + M_PI / 2.f - droneRadStep);
	float droneRightTopX = DRONE_START_X + DRONE_HALF_DIAGONAL * cos(-M_PI / 2.f + JET_ANGLE + droneRadStep);
	float droneRightTopY = DRONE_START_Y + DRONE_HALF_DIAGONAL * sin(-M_PI / 2.f + JET_ANGLE + droneRadStep);
	float droneRightBottomX = DRONE_START_X + DRONE_HALF_DIAGONAL * cos(-M_PI / 2.f + JET_ANGLE - droneRadStep);
	float droneRightBottomY = DRONE_START_Y + DRONE_HALF_DIAGONAL * sin(-M_PI / 2.f + JET_ANGLE - droneRadStep);
	glBegin(GL_POLYGON);
	glColor3d(0, 0, 0);
	glVertex2i(droneLeftBottomX, droneLeftBottomY);
	glVertex2i(droneLeftTopX, droneLeftTopY);
	glVertex2i(droneRightTopX, droneRightTopY);
	glVertex2i(droneRightBottomX, droneRightBottomY);
	glEnd();
	// draw rocket
	for (auto& face : faces)
	{
		glBegin(GL_LINE_STRIP);
		glColor3d(0.f, 1.f, 0.f);
		for (auto& id : face)
		{
			std::vector<float> vertice = rocket[id - 1];
			glVertex2i(vertice[0] + DRONE_START_X, vertice[1] + DRONE_START_Y);
		}
		glEnd();
	}
	// jet 1 coords
	float A1_X = droneLeftTopX;
	float A1_Y = droneLeftTopY;
	float B1_X = A1_X - DRONE_HALF_WIDTH / 2.f * cos(THETA) - JET_LEN1 * sin(THETA);
	float B1_Y = A1_Y + DRONE_HALF_WIDTH / 2.f * sin(THETA) - JET_LEN1 * cos(THETA);
	float C1_X = A1_X + DRONE_HALF_WIDTH / 2.f * cos(THETA) - JET_LEN1 * sin(THETA);
	float C1_Y = A1_Y - DRONE_HALF_WIDTH / 2.f * sin(THETA) - JET_LEN1 * cos(THETA);
	// jet 2 coords
	float A2_X = droneRightTopX;
	float A2_Y = droneRightTopY;
	float B2_X = A2_X - DRONE_HALF_WIDTH / 2.f * cos(THETA) - JET_LEN2 * sin(THETA);
	float B2_Y = A2_Y + DRONE_HALF_WIDTH / 2.f * sin(THETA) - JET_LEN2 * cos(THETA);
	float C2_X = A2_X + DRONE_HALF_WIDTH / 2.f * cos(THETA) - JET_LEN2 * sin(THETA);
	float C2_Y = A2_Y - DRONE_HALF_WIDTH / 2.f * sin(THETA) - JET_LEN2 * cos(THETA);
	// draw jet 1
	glBegin(GL_TRIANGLES);
	glColor3d(1, 0.5f, 0);
	glVertex2i(A1_X, A1_Y);
	glVertex2i(B1_X, B1_Y);
	glVertex2i(C1_X, C1_Y);
	glEnd();
	// draw jet 2
	glBegin(GL_TRIANGLES);
	glColor3d(1, 0.5f, 0);
	glVertex2i(A2_X, A2_Y);
	glVertex2i(B2_X, B2_Y);
	glVertex2i(C2_X, C2_Y);
	glEnd();
	// draw target
	glLineWidth(3);
	glBegin(GL_LINES);
	glColor3d(1, 0, 0);
	glVertex2i(-15, 0);
	glVertex2i(15, 0);
	glVertex2i(0, 15);
	glVertex2i(0, -15);

	glEnd();
	glFlush();
}


CriticNet CRITIC_NETWORK_GUI;
void RunDrone()
{
	auto currentTime = std::chrono::high_resolution_clock::now();
	if ((currentTime - startTime).count() * 1e-7 > -1.f)
	{
		auto action = ActionsWithoutActor(droneStateGUI, CRITIC_NETWORK_GUI);
		auto Q = CRITIC_NETWORK_GUI->forwardNoGrad(torch::cat({ droneStateGUI, action }, 1));
		auto newState = UpdatePhysics(droneStateGUI, action);
		droneStateGUI = newState;
		actionGUI = action * MAX_FORCE;
		glutPostRedisplay();
		startTime = currentTime;
	}
}


void PassiveMotion(int x, int y)
{
	NEW_MOUSE_X = x;
	NEW_MOUSE_Y = glutGet(GLUT_WINDOW_HEIGHT) - y;
}


void GUI(int argc, char** argv)
{
	// INIT FIELD
	glutInit(&argc, argv);
	glutInitWindowPosition(0, 0); // top right corner
	glutInitWindowSize(500, 500); // width, height
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
	glutCreateWindow("field");

	torch::load(CRITIC_NETWORK_GUI, "criticModelBest.pt");
	CRITIC_NETWORK_GUI->to(device);

	MyInitField();
	glutDisplayFunc(Draw);
	glutIdleFunc(RunDrone);
	glutPassiveMotionFunc(PassiveMotion);
	glutMainLoop();
}


int main(int argc, char** argv)
{
	GUI(argc, argv);
}