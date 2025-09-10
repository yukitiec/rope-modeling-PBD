#pragma once
#include <vector>
#include <Eigen/Dense>
#include "global_param.h"

// Constraint class for PBD
class Constraint {
public:
    int i; // First particle index
    int j; // Second particle index
    float d; // Rest length

    Constraint(int i, int j, float d) : i(i), j(j), d(d) {}
};

// Simple 3x3 matrix class for velocity damping
class SquareMatrix {
public:
    float data[3][3];

    SquareMatrix() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                data[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    float& operator()(int i, int j) { return data[i][j]; }
    const float& operator()(int i, int j) const { return data[i][j]; }

    SquareMatrix operator+(const SquareMatrix& other) const {
        SquareMatrix result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    SquareMatrix operator*(const SquareMatrix& other) const {
        SquareMatrix result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.data[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    SquareMatrix operator*(float scalar) const {
        SquareMatrix result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    SquareMatrix Transpose() const {
        SquareMatrix result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.data[i][j] = data[j][i];
            }
        }
        return result;
    }

    SquareMatrix Inverse() const {
        Eigen::Matrix3f eigenMatrix;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                eigenMatrix(i, j) = data[i][j];
            }
        }

        Eigen::Matrix3f inverse = eigenMatrix.inverse();
        SquareMatrix result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.data[i][j] = inverse(i, j);
            }
        }
        return result;
    }
};

// PBD Rope Simulation class
class PBDRope {
public:
    // Simulation parameters
    int n = 24; // Number of particles
    float k = 0.5f; // Stiffness
    float dt = 0.01f; // Delta time
    Eigen::Vector3f gravity = Eigen::Vector3f(0.0f, -9.81f, 0.0f); // Gravity
    float kDamping = 0.03f; // Velocity damping constant

    // Particle data
    std::vector<Eigen::Vector3f> x; // Particle positions
    std::vector<Eigen::Vector3f> v; // Particle velocities
    std::vector<Eigen::Vector3f> p; // Predicted positions
    std::vector<bool> isFixed; // Fixed particles
    std::vector<float> m; // Masses

    // Constraints
    std::vector<Constraint> constraints;

    // Start and end points
    Eigen::Vector3f startPoint;
    Eigen::Vector3f endPoint;

public:
    PBDRope();
    PBDRope(int numParticles, const Eigen::Vector3f& start, const Eigen::Vector3f& end);

    void initialize();
    void update();
    void velocityDamping();
    std::vector<Eigen::Vector3f> setParticle(); // Returns updated particle positions
    std::vector<Eigen::Vector3f> getParticlePositions() const; // Get current particle positions
    std::vector<Eigen::Vector3f> getParticleVelocities() const; // Get current particle velocities

private:
    void initializeParticles();
    void initializeConstraints();
};
