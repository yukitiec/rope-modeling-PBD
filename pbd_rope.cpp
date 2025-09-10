#include "pbd_rope.h"
#include <iostream>

// Default constructor
PBDRope::PBDRope() : startPoint(0.0f, 5.0f, 0.0f), endPoint(0.0f, -5.0f, 0.0f) {
    initialize();
}

// Constructor with custom parameters
PBDRope::PBDRope(int numParticles, const Eigen::Vector3f& start, const Eigen::Vector3f& end)
    : n(numParticles), startPoint(start), endPoint(end) {
    initialize();
}

// Initialize the rope simulation
void PBDRope::initialize() {
    initializeParticles();
    initializeConstraints();
}

// Initialize particles
void PBDRope::initializeParticles() {
    // Resize vectors
    p.resize(n); // Predicted positions
    x.resize(n); // Positions
    v.resize(n); // Velocities
    m.resize(n); // Masses
    isFixed.resize(n); // Fixed particles

    // Initialize particles
    for (int i = 0; i < n; i++) {
        float t = static_cast<float>(i) / (n - 1);

        // Position - linear interpolation between start and end points
        x[i] = startPoint + t * (endPoint - startPoint);

        // Velocity - initially zero
        v[i] = Eigen::Vector3f::Zero();

        // Mass - uniform mass distribution
        m[i] = 1.0f;
    }

    // Fix the endpoints
    isFixed[0] = true;
    isFixed[n - 1] = true;
}

// Initialize constraints
void PBDRope::initializeConstraints() {
    constraints.clear();
    constraints.reserve(n - 1);

    for (int i = 0; i < n - 1; i++) {
        // Calculate rest length
        float d = (x[i] - x[i + 1]).norm();

        // Connect adjacent particles
        constraints.emplace_back(i, i + 1, d);
    }
}

// Main update function based on the provided C# code
void PBDRope::update() {
    // External force velocity change
    for (int i = 0; i < n; i++) {
        v[i] += gravity * dt;
        if (isFixed[i]) v[i] = Eigen::Vector3f::Zero();
    }

    // Velocity Damping
    velocityDamping();

    // Position update
    for (int i = 0; i < n; i++) {
        p[i] = x[i] + v[i] * dt;
        x[i] = p[i];
    }

    // Fix endpoints
    x[0] = startPoint;
    x[n - 1] = endPoint;

    // Velocity update (Project Constraints)
    for (int i = 0; i < constraints.size(); i++) {
        const auto& c = constraints[i];
        const auto& p1 = p[c.i];
        const auto& p2 = p[c.j];

        float w1 = 1.0f / m[c.i]; // Inverse mass
        float w2 = 1.0f / m[c.j]; // Inverse mass

        float diff = (p1 - p2).norm();
        if (diff > 0.0f) {
            Eigen::Vector3f direction = (p1 - p2).normalized();

            Eigen::Vector3f dp1 = -k * w1 / (w1 + w2) * (diff - c.d) * direction;
            Eigen::Vector3f dp2 = k * w2 / (w1 + w2) * (diff - c.d) * direction;

            v[c.i] += dp1 / dt;
            v[c.j] += dp2 / dt;
        }
    }

    // Apply to particles
    setParticle();
}

// Velocity damping function based on the provided C# code
void PBDRope::velocityDamping() {
    Eigen::Vector3f xcm = Eigen::Vector3f::Zero();
    Eigen::Vector3f vcm = Eigen::Vector3f::Zero();
    float totalMass = 0.0f;

    for (int i = 0; i < n; i++) {
        xcm += x[i];
        vcm += v[i];
        totalMass += m[i];
    }
    xcm /= totalMass;
    vcm /= totalMass;

    Eigen::Vector3f L = Eigen::Vector3f::Zero();
    SquareMatrix I;
    std::vector<Eigen::Vector3f> rs(n);

    for (int i = 0; i < n; i++) {
        Eigen::Vector3f r = x[i] - xcm;
        rs[i] = r;

        SquareMatrix R;
        R(0, 1) = r(2);
        R(0, 2) = -r(1);
        R(1, 0) = -r(2);
        R(1, 2) = r(0);
        R(2, 0) = r(1);
        R(2, 1) = -r(0);

        L += r.cross(m[i] * v[i]);
        I = I + (R * R.Transpose()) * m[i];
    }

    Eigen::Vector3f omega = Eigen::Vector3f::Zero();
    try {
        SquareMatrix I_inv = I.Inverse();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                omega(i) += I_inv(i, j) * L(j);
            }
        }
    } catch (...) {
        // If matrix inversion fails, skip damping
        return;
    }

    for (int i = 0; i < n; i++) {
        Eigen::Vector3f deltaV = vcm + omega.cross(rs[i]) - v[i];
        v[i] += kDamping * deltaV;
    }
}

// Return updated particle positions
std::vector<Eigen::Vector3f> PBDRope::setParticle() {
    // Return a copy of the current particle positions
    return x;
}

// Get current particle positions
std::vector<Eigen::Vector3f> PBDRope::getParticlePositions() const {
    return x;
}

// Get current particle velocities
std::vector<Eigen::Vector3f> PBDRope::getParticleVelocities() const {
    return v;
}
