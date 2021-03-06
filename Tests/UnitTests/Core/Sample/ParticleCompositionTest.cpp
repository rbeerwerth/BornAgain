#include "ParticleComposition.h"
#include "BornAgainNamespace.h"
#include "FormFactorFullSphere.h"
#include "MaterialFactoryFuncs.h"
#include "MathConstants.h"
#include "Particle.h"
#include "google_test.h"

class ParticleCompositionTest : public ::testing::Test
{
protected:
    ~ParticleCompositionTest();
};

ParticleCompositionTest::~ParticleCompositionTest() = default;

TEST_F(ParticleCompositionTest, ParticleCompositionDefaultConstructor)
{
    std::unique_ptr<ParticleComposition> composition(new ParticleComposition());
    std::vector<kvector_t> positions;
    positions.push_back(kvector_t(0.0, 0.0, 0.0));
    EXPECT_EQ(BornAgain::ParticleCompositionType, composition->getName());
    EXPECT_EQ(0u, composition->nbrParticles());
}

TEST_F(ParticleCompositionTest, ParticleCompositionClone)
{
    ParticleComposition composition;
    Particle particle;
    kvector_t position = kvector_t(1.0, 1.0, 1.0);
    Material material = HomogeneousMaterial("Air", 0.0, 0.0);
    composition.addParticle(particle, position);

    std::unique_ptr<ParticleComposition> clone(composition.clone());

    EXPECT_EQ(clone->getName(), composition.getName());
    std::vector<const INode*> children = clone->getChildren();
    EXPECT_EQ(children.size(), 1u);
    auto p_particle = dynamic_cast<const IParticle*>(children[0]);

    EXPECT_EQ(p_particle->getName(), particle.getName());
    EXPECT_EQ(p_particle->rotation(), nullptr);
    EXPECT_EQ(p_particle->position(), position);
}

TEST_F(ParticleCompositionTest, getChildren)
{
    Material material = HomogeneousMaterial("Air", 0.0, 0.0);

    ParticleComposition composition;
    composition.addParticle(Particle(material, FormFactorFullSphere(1.0)));
    composition.addParticle(Particle(material, FormFactorFullSphere(1.0)));
    composition.setRotation(RotationY(45.));

    std::vector<const INode*> children = composition.getChildren();
    EXPECT_EQ(children.size(), 3u);
    EXPECT_EQ(children.at(0)->getName(), BornAgain::YRotationType);
    EXPECT_EQ(children.at(1)->getName(), BornAgain::ParticleType);
    EXPECT_EQ(children.at(2)->getName(), BornAgain::ParticleType);
}
