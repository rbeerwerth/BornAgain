
// File: index.xml

// File: classAdjustMinimizerStrategy.xml
%feature("docstring") AdjustMinimizerStrategy "

Strategy modifies mimimizer settings before running minimization round.

C++ includes: AdjustMinimizerStrategy.h
";

%feature("docstring")  AdjustMinimizerStrategy::AdjustMinimizerStrategy "AdjustMinimizerStrategy::AdjustMinimizerStrategy()
";

%feature("docstring")  AdjustMinimizerStrategy::AdjustMinimizerStrategy "AdjustMinimizerStrategy::AdjustMinimizerStrategy(const std::string &minimizerName, const std::string &algorithmName=\"\", const std::string &optionString=\"\")
";

%feature("docstring")  AdjustMinimizerStrategy::clone "AdjustMinimizerStrategy * AdjustMinimizerStrategy::clone() const 
";

%feature("docstring")  AdjustMinimizerStrategy::setMinimizer "void AdjustMinimizerStrategy::setMinimizer(const std::string &minimizerName, const std::string &algorithmName=\"\", const std::string &optionString=\"\")
";

%feature("docstring")  AdjustMinimizerStrategy::execute "void AdjustMinimizerStrategy::execute()
";


// File: classBasic2DLatticeBuilder.xml
%feature("docstring") Basic2DLatticeBuilder "

Builds sample: 2D lattice with arbitrary angle and different lattice length_1 and length_2.

C++ includes: TwoDimLatticeBuilder.h
";

%feature("docstring")  Basic2DLatticeBuilder::Basic2DLatticeBuilder "Basic2DLatticeBuilder::Basic2DLatticeBuilder()
";

%feature("docstring")  Basic2DLatticeBuilder::buildSample "MultiLayer * Basic2DLatticeBuilder::buildSample() const 
";


// File: classBasic2DParaCrystalBuilder.xml
%feature("docstring") Basic2DParaCrystalBuilder "

Builds sample: basic two dimensional paracrystal with various probability distribution functions (PDF's). They are initialized via component service.

C++ includes: ParaCrystalBuilder.h
";

%feature("docstring")  Basic2DParaCrystalBuilder::Basic2DParaCrystalBuilder "Basic2DParaCrystalBuilder::Basic2DParaCrystalBuilder()
";

%feature("docstring")  Basic2DParaCrystalBuilder::~Basic2DParaCrystalBuilder "Basic2DParaCrystalBuilder::~Basic2DParaCrystalBuilder()
";

%feature("docstring")  Basic2DParaCrystalBuilder::buildSample "MultiLayer * Basic2DParaCrystalBuilder::buildSample() const 
";


// File: classBasicLattice.xml
%feature("docstring") BasicLattice "";

%feature("docstring")  BasicLattice::BasicLattice "BasicLattice::BasicLattice(double length1, double length2, double angle, double rotation_angle=0.0)
";

%feature("docstring")  BasicLattice::clone "BasicLattice * BasicLattice::clone() const 
";

%feature("docstring")  BasicLattice::accept "void BasicLattice::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";


// File: classBasicVector3D.xml
%feature("docstring") BasicVector3D "

Three-dimensional vector template, for use with integer, double, or complex components.

C++ includes: BasicVector3D.h
";

%feature("docstring")  BasicVector3D::BasicVector3D "BasicVector3D< T >::BasicVector3D()

Default constructor. 
";

%feature("docstring")  BasicVector3D::BasicVector3D "BasicVector3D< T >::BasicVector3D(const T x1, const T y1, const T z1)

Constructor from cartesian components. 
";

%feature("docstring")  BasicVector3D::x "T BasicVector3D< T >::x() const

Returns x-component in cartesian coordinate system. 
";

%feature("docstring")  BasicVector3D::y "T BasicVector3D< T >::y() const

Returns y-component in cartesian coordinate system. 
";

%feature("docstring")  BasicVector3D::z "T BasicVector3D< T >::z() const

Returns z-component in cartesian coordinate system. 
";

%feature("docstring")  BasicVector3D::setX "void BasicVector3D< T >::setX(const T &a)

Sets x-component in cartesian coordinate system. 
";

%feature("docstring")  BasicVector3D::setY "void BasicVector3D< T >::setY(const T &a)

Sets y-component in cartesian coordinate system. 
";

%feature("docstring")  BasicVector3D::setZ "void BasicVector3D< T >::setZ(const T &a)

Sets z-component in cartesian coordinate system. 
";

%feature("docstring")  BasicVector3D::conj "BasicVector3D<T> BasicVector3D< T >::conj() const

Returns complex conjugate vector. 
";

%feature("docstring")  BasicVector3D::mag2 "double BasicVector3D< T >::mag2() const

Returns squared magnitude squared of the vector. 
";

%feature("docstring")  BasicVector3D::mag "double BasicVector3D< T >::mag() const

Returns magnitude of the vector. 
";

%feature("docstring")  BasicVector3D::magxy2 "double BasicVector3D< T >::magxy2() const

Returns squared distance from z axis. 
";

%feature("docstring")  BasicVector3D::magxy "double BasicVector3D< T >::magxy() const

Returns distance from z axis. 
";

%feature("docstring")  BasicVector3D::phi "double BasicVector3D< T >::phi() const

Returns azimuth angle. 
";

%feature("docstring")  BasicVector3D::theta "double BasicVector3D< T >::theta() const

Returns polar angle. 
";

%feature("docstring")  BasicVector3D::cosTheta "double BasicVector3D< T >::cosTheta() const

Returns cosine of polar angle. 
";

%feature("docstring")  BasicVector3D::sin2Theta "double BasicVector3D< T >::sin2Theta() const

Returns squared sine of polar angle. 
";

%feature("docstring")  BasicVector3D::unit "BasicVector3D<T> BasicVector3D< T >::unit() const

Returns unit vector in direction of this. Throws for null vector. 
";

%feature("docstring")  BasicVector3D::complex "BasicVector3D<std::complex<double> > BasicVector3D< T >::complex() const

Returns this, trivially converted to complex type. 
";

%feature("docstring")  BasicVector3D::real "BasicVector3D<double> BasicVector3D< T >::real() const

Returns real parts. 
";

%feature("docstring")  BasicVector3D::dot "auto BasicVector3D< T >::dot(const BasicVector3D< U > &v) const -> decltype(this->x()*v.x())

Returns dot product of vectors (antilinear in the first [=self] argument).

Returns dot product of (complex) vectors (antilinear in the first [=self] argument). 
";

%feature("docstring")  BasicVector3D::cross "auto BasicVector3D< T >::cross(const BasicVector3D< U > &v) const -> BasicVector3D< decltype(this->x()*v.x())>

Returns cross product of vectors (linear in both arguments).

Returns cross product of (complex) vectors. 
";

%feature("docstring")  BasicVector3D::angle "double BasicVector3D< T >::angle(const BasicVector3D< T > &v) const

Returns angle with respect to another vector. 
";

%feature("docstring")  BasicVector3D::project "BasicVector3D<T> BasicVector3D< T >::project(const BasicVector3D< T > &v) const

Returns projection of this onto other vector: (this*v)*v/|v|^2. 
";

%feature("docstring")  BasicVector3D::rotatedX "BasicVector3D<T> BasicVector3D< T >::rotatedX(double a) const

Returns result of rotation around x-axis. 
";

%feature("docstring")  BasicVector3D::rotatedY "BasicVector3D<T> BasicVector3D< T >::rotatedY(double a) const

Returns result of rotation around y-axis. 
";

%feature("docstring")  BasicVector3D::rotatedZ "BasicVector3D<T> BasicVector3D< T >::rotatedZ(double a) const

Returns result of rotation around z-axis. 
";

%feature("docstring")  BasicVector3D::rotated "BasicVector3D<T> BasicVector3D< T >::rotated(double a, const BasicVector3D< T > &v) const

Returns result of rotation around the axis specified by another vector. 
";

%feature("docstring")  BasicVector3D::conj "BasicVector3D< complex_t > BasicVector3D< complex_t >::conj() const

Returns complex conjugate vector. 
";

%feature("docstring")  BasicVector3D::real "BasicVector3D< double > BasicVector3D< complex_t >::real() const

Returns real parts. 
";

%feature("docstring")  BasicVector3D::unit "BasicVector3D< complex_t > BasicVector3D< complex_t >::unit() const
";


// File: classBeam.xml
%feature("docstring") Beam "

Ideal collimated beam defined by wavelength, direction and intensity.

C++ includes: Beam.h
";

%feature("docstring")  Beam::Beam "Beam::Beam()
";

%feature("docstring")  Beam::Beam "Beam::Beam(const Beam &other)
";

%feature("docstring")  Beam::~Beam "virtual Beam::~Beam()
";

%feature("docstring")  Beam::getCentralK "kvector_t Beam::getCentralK() const

Returns the wavevector. 
";

%feature("docstring")  Beam::setCentralK "void Beam::setCentralK(double wavelength, double alpha_i, double phi_i)

Sets the wavevector in terms of wavelength and incoming angles. 
";

%feature("docstring")  Beam::getIntensity "double Beam::getIntensity() const

Returns the beam intensity in neutrons/sec. 
";

%feature("docstring")  Beam::setIntensity "void Beam::setIntensity(double intensity)

Sets the beam intensity in neutrons/sec. 
";

%feature("docstring")  Beam::setPolarization "void Beam::setPolarization(const kvector_t bloch_vector)

Sets the polarization density matrix according to the given Bloch vector. 
";

%feature("docstring")  Beam::getPolarization "Eigen::Matrix2cd Beam::getPolarization() const

Returns the polarization density matrix (in spin basis along z-axis) 
";

%feature("docstring")  Beam::getWavelength "double Beam::getWavelength() const 
";

%feature("docstring")  Beam::getAlpha "double Beam::getAlpha() const 
";

%feature("docstring")  Beam::getPhi "double Beam::getPhi() const 
";

%feature("docstring")  Beam::accept "void Beam::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";


// File: structBin1D.xml
%feature("docstring") Bin1D "";

%feature("docstring")  Bin1D::Bin1D "Bin1D::Bin1D()
";

%feature("docstring")  Bin1D::Bin1D "Bin1D::Bin1D(double lower, double upper)
";

%feature("docstring")  Bin1D::getMidPoint "double Bin1D::getMidPoint() const 
";

%feature("docstring")  Bin1D::getBinSize "double Bin1D::getBinSize() const 
";


// File: classBin1DCVector.xml
%feature("docstring") Bin1DCVector "

An one-dimensional range of cvector_t's.

C++ includes: Bin.h
";

%feature("docstring")  Bin1DCVector::Bin1DCVector "Bin1DCVector::Bin1DCVector()
";

%feature("docstring")  Bin1DCVector::Bin1DCVector "Bin1DCVector::Bin1DCVector(const cvector_t lower, const cvector_t upper)
";

%feature("docstring")  Bin1DCVector::Bin1DCVector "Bin1DCVector::Bin1DCVector(double wavelength, const Bin1D &alpha_bin, const Bin1D &phi_bin)

creation on  Bin1DCVector from alpha and phi bins 
";

%feature("docstring")  Bin1DCVector::getMidPoint "cvector_t Bin1DCVector::getMidPoint() const 
";

%feature("docstring")  Bin1DCVector::getDelta "cvector_t Bin1DCVector::getDelta() const 
";


// File: structBin1DKVector.xml
%feature("docstring") Bin1DKVector "

An one-dimensional range of kvector_t's.

C++ includes: Bin.h
";

%feature("docstring")  Bin1DKVector::Bin1DKVector "Bin1DKVector::Bin1DKVector()
";

%feature("docstring")  Bin1DKVector::Bin1DKVector "Bin1DKVector::Bin1DKVector(const kvector_t lower, const kvector_t upper)
";

%feature("docstring")  Bin1DKVector::Bin1DKVector "Bin1DKVector::Bin1DKVector(double wavelength, const Bin1D &alpha_bin, const Bin1D &phi_bin)

creation on  Bin1DKVector from alpha and phi bins 
";

%feature("docstring")  Bin1DKVector::getMidPoint "kvector_t Bin1DKVector::getMidPoint() const 
";

%feature("docstring")  Bin1DKVector::getDelta "kvector_t Bin1DKVector::getDelta() const 
";


// File: classBoxCompositionBuilder.xml
%feature("docstring") BoxCompositionBuilder "

Parent class to test all kind of compositions made out of boxes. Reproduces Python functional test transform_BoxComposition.py

C++ includes: BoxCompositionBuilder.h
";

%feature("docstring")  BoxCompositionBuilder::BoxCompositionBuilder "BoxCompositionBuilder::BoxCompositionBuilder()
";

%feature("docstring")  BoxCompositionBuilder::~BoxCompositionBuilder "BoxCompositionBuilder::~BoxCompositionBuilder()
";


// File: classBoxCompositionRotateXBuilder.xml
%feature("docstring") BoxCompositionRotateXBuilder "

Two boxes in particle composition rotated in X by 90 degrees.

C++ includes: BoxCompositionBuilder.h
";

%feature("docstring")  BoxCompositionRotateXBuilder::BoxCompositionRotateXBuilder "BoxCompositionRotateXBuilder::BoxCompositionRotateXBuilder()
";

%feature("docstring")  BoxCompositionRotateXBuilder::buildSample "MultiLayer * BoxCompositionRotateXBuilder::buildSample() const 
";


// File: classBoxCompositionRotateYBuilder.xml
%feature("docstring") BoxCompositionRotateYBuilder "

Two boxes in particle composition rotated in Y by 90 degrees.

C++ includes: BoxCompositionBuilder.h
";

%feature("docstring")  BoxCompositionRotateYBuilder::BoxCompositionRotateYBuilder "BoxCompositionRotateYBuilder::BoxCompositionRotateYBuilder()
";

%feature("docstring")  BoxCompositionRotateYBuilder::buildSample "MultiLayer * BoxCompositionRotateYBuilder::buildSample() const 
";


// File: classBoxCompositionRotateZandYBuilder.xml
%feature("docstring") BoxCompositionRotateZandYBuilder "

Two boxes in particle composition rotated in Z and Y by 90 degrees.

C++ includes: BoxCompositionBuilder.h
";

%feature("docstring")  BoxCompositionRotateZandYBuilder::BoxCompositionRotateZandYBuilder "BoxCompositionRotateZandYBuilder::BoxCompositionRotateZandYBuilder()
";

%feature("docstring")  BoxCompositionRotateZandYBuilder::buildSample "MultiLayer * BoxCompositionRotateZandYBuilder::buildSample() const 
";


// File: classBoxCompositionRotateZBuilder.xml
%feature("docstring") BoxCompositionRotateZBuilder "

Two boxes in particle composition rotated in Z by 90 degrees.

C++ includes: BoxCompositionBuilder.h
";

%feature("docstring")  BoxCompositionRotateZBuilder::BoxCompositionRotateZBuilder "BoxCompositionRotateZBuilder::BoxCompositionRotateZBuilder()
";

%feature("docstring")  BoxCompositionRotateZBuilder::buildSample "MultiLayer * BoxCompositionRotateZBuilder::buildSample() const 
";


// File: classBoxStackCompositionBuilder.xml
%feature("docstring") BoxStackCompositionBuilder "

Two different boxes are first rotated and then composed, composition is then rotated.

C++ includes: BoxCompositionBuilder.h
";

%feature("docstring")  BoxStackCompositionBuilder::BoxStackCompositionBuilder "BoxStackCompositionBuilder::BoxStackCompositionBuilder()
";

%feature("docstring")  BoxStackCompositionBuilder::buildSample "MultiLayer * BoxStackCompositionBuilder::buildSample() const 
";


// File: structIntegratorReal_1_1CallBackHolder.xml
%feature("docstring") IntegratorReal::CallBackHolder "

structure holding the object and possible extra parameters

C++ includes: IntegratorReal.h
";


// File: structIntegratorMCMiser_1_1CallBackHolder.xml
%feature("docstring") IntegratorMCMiser::CallBackHolder "

structure holding the object and possible extra parameters

C++ includes: IntegratorMCMiser.h
";


// File: classCenteredSquareLatticeBuilder.xml
%feature("docstring") CenteredSquareLatticeBuilder "

Builds sample: 2D lattice with different disorder (IsGISAXS example #6).

C++ includes: TwoDimLatticeBuilder.h
";

%feature("docstring")  CenteredSquareLatticeBuilder::CenteredSquareLatticeBuilder "CenteredSquareLatticeBuilder::CenteredSquareLatticeBuilder()
";

%feature("docstring")  CenteredSquareLatticeBuilder::buildSample "MultiLayer * CenteredSquareLatticeBuilder::buildSample() const 
";


// File: classChiSquaredModule.xml
%feature("docstring") ChiSquaredModule "

Calculation of chi2 between two data sets.

C++ includes: ChiSquaredModule.h
";

%feature("docstring")  ChiSquaredModule::ChiSquaredModule "ChiSquaredModule::ChiSquaredModule()
";

%feature("docstring")  ChiSquaredModule::ChiSquaredModule "ChiSquaredModule::ChiSquaredModule(const ChiSquaredModule &other)
";

%feature("docstring")  ChiSquaredModule::~ChiSquaredModule "virtual ChiSquaredModule::~ChiSquaredModule()
";

%feature("docstring")  ChiSquaredModule::clone "virtual ChiSquaredModule* ChiSquaredModule::clone() const

clone method 
";

%feature("docstring")  ChiSquaredModule::processFitElements "void ChiSquaredModule::processFitElements(std::vector< FitElement >::iterator first, std::vector< FitElement >::iterator last)
";


// File: classExceptions_1_1ClassInitializationException.xml
%feature("docstring") Exceptions::ClassInitializationException "";

%feature("docstring")  Exceptions::ClassInitializationException::ClassInitializationException "Exceptions::ClassInitializationException::ClassInitializationException(const std::string &message)
";


// File: classComputationStatus.xml
%feature("docstring") ComputationStatus "

Completion status (flag and text) of a numeric computation.

C++ includes: ComputationStatus.h
";

%feature("docstring")  ComputationStatus::ComputationStatus "ComputationStatus::ComputationStatus()
";

%feature("docstring")  ComputationStatus::isCompleted "bool ComputationStatus::isCompleted() const 
";

%feature("docstring")  ComputationStatus::errorMessage "std::string ComputationStatus::errorMessage() const 
";

%feature("docstring")  ComputationStatus::setRunning "void ComputationStatus::setRunning()
";

%feature("docstring")  ComputationStatus::setCompleted "void ComputationStatus::setCompleted()
";

%feature("docstring")  ComputationStatus::setFailed "void ComputationStatus::setFailed()
";

%feature("docstring")  ComputationStatus::setErrorMessage "void ComputationStatus::setErrorMessage(const std::string &message)
";


// File: classConstKBinAxis.xml
%feature("docstring") ConstKBinAxis "

Axis with fixed bin size in sin(angle) space.

C++ includes: ConstKBinAxis.h
";

%feature("docstring")  ConstKBinAxis::ConstKBinAxis "ConstKBinAxis::ConstKBinAxis(const std::string &name, size_t nbins, double start, double end)

ConstKBinAxis constructor.

Parameters:
-----------

name: 
Axis name

nbins: 
number of bins

start: 
low edge of first bin

end: 
upper edge of last bin 
";

%feature("docstring")  ConstKBinAxis::~ConstKBinAxis "ConstKBinAxis::~ConstKBinAxis() final
";

%feature("docstring")  ConstKBinAxis::clone "ConstKBinAxis * ConstKBinAxis::clone() const final

clone function 
";

%feature("docstring")  ConstKBinAxis::createClippedAxis "ConstKBinAxis * ConstKBinAxis::createClippedAxis(double left, double right) const final

Creates a new clipped axis. 
";


// File: classConvolutionDetectorResolution.xml
%feature("docstring") ConvolutionDetectorResolution "

Limitation: this class assumes that the data points are evenly distributed on each axis.

Convolutes the intensity in 1 or 2 dimensions with a resolution function.

C++ includes: ConvolutionDetectorResolution.h
";

%feature("docstring")  ConvolutionDetectorResolution::ConvolutionDetectorResolution "ConvolutionDetectorResolution::ConvolutionDetectorResolution(cumulative_DF_1d res_function_1d)

Constructor taking a 1 dimensional resolution function as argument. 
";

%feature("docstring")  ConvolutionDetectorResolution::ConvolutionDetectorResolution "ConvolutionDetectorResolution::ConvolutionDetectorResolution(const IResolutionFunction2D &p_res_function_2d)

Constructor taking a 2 dimensional resolution function as argument. 
";

%feature("docstring")  ConvolutionDetectorResolution::~ConvolutionDetectorResolution "ConvolutionDetectorResolution::~ConvolutionDetectorResolution()
";

%feature("docstring")  ConvolutionDetectorResolution::applyDetectorResolution "void ConvolutionDetectorResolution::applyDetectorResolution(OutputData< double > *p_intensity_map) const

Convolve given intensities with the encapsulated resolution. 
";

%feature("docstring")  ConvolutionDetectorResolution::clone "ConvolutionDetectorResolution * ConvolutionDetectorResolution::clone() const 
";

%feature("docstring")  ConvolutionDetectorResolution::accept "void ConvolutionDetectorResolution::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  ConvolutionDetectorResolution::getResolutionFunction2D "const IResolutionFunction2D * ConvolutionDetectorResolution::getResolutionFunction2D() const 
";

%feature("docstring")  ConvolutionDetectorResolution::getChildren "std::vector< const INode * > ConvolutionDetectorResolution::getChildren() const

Returns a vector of children (const). 
";


// File: classConvolve.xml
%feature("docstring") Convolve "

Convolution of two real vectors (in 1D or 2D) using Fast Fourier Transform.

Usage: std::vector<double> signal, kernel, result;  Convolve cv; cv.fftconvolve(signal, kernel, result)

Given code rely on code from Jeremy Fix page, http://jeremy.fix.free.fr/spip.php?article15, see also \"Efficient convolution using the Fast Fourier Transform, Application in C++\" by Jeremy Fix, May 30, 2011

C++ includes: Convolve.h
";

%feature("docstring")  Convolve::Convolve "Convolve::Convolve()
";

%feature("docstring")  Convolve::fftconvolve "void Convolve::fftconvolve(const double1d_t &source, const double1d_t &kernel, double1d_t &result)

convolution in 1D 
";

%feature("docstring")  Convolve::fftconvolve "void Convolve::fftconvolve(const double2d_t &source, const double2d_t &kernel, double2d_t &result)

convolution in 2D 
";

%feature("docstring")  Convolve::init "void Convolve::init(int h_src, int w_src, int h_kernel, int w_kernel)

prepare arrays for 2D convolution of given vectors 
";

%feature("docstring")  Convolve::setMode "void Convolve::setMode(EConvolutionMode mode)

Sets convolution mode. 
";


// File: classCoreShellBoxRotateZandYBuilder.xml
%feature("docstring") CoreShellBoxRotateZandYBuilder "

Rotation and translation of core shell box particle in 3 layers system.

C++ includes: CoreShellParticleBuilder.h
";

%feature("docstring")  CoreShellBoxRotateZandYBuilder::CoreShellBoxRotateZandYBuilder "CoreShellBoxRotateZandYBuilder::CoreShellBoxRotateZandYBuilder()
";

%feature("docstring")  CoreShellBoxRotateZandYBuilder::buildSample "MultiLayer * CoreShellBoxRotateZandYBuilder::buildSample() const 
";


// File: classCoreShellParticleBuilder.xml
%feature("docstring") CoreShellParticleBuilder "

Builds sample: Core Shell Nanoparticles (IsGISAXS example #11).

C++ includes: CoreShellParticleBuilder.h
";

%feature("docstring")  CoreShellParticleBuilder::CoreShellParticleBuilder "CoreShellParticleBuilder::CoreShellParticleBuilder()
";

%feature("docstring")  CoreShellParticleBuilder::buildSample "MultiLayer * CoreShellParticleBuilder::buildSample() const 
";


// File: classCosineRippleBuilder.xml
%feature("docstring") CosineRippleBuilder "

Builds sample: cosine ripple within the 1D-paracrystal model.

C++ includes: RipplesBuilder.h
";

%feature("docstring")  CosineRippleBuilder::CosineRippleBuilder "CosineRippleBuilder::CosineRippleBuilder()
";

%feature("docstring")  CosineRippleBuilder::buildSample "MultiLayer * CosineRippleBuilder::buildSample() const 
";


// File: classCrystal.xml
%feature("docstring") Crystal "

A crystal structure with a  ParticleComposition as a basis. Used in  MesoCrystal, where it is given an outer shape.

C++ includes: Crystal.h
";

%feature("docstring")  Crystal::Crystal "Crystal::Crystal(const ParticleComposition &lattice_basis, const Lattice &lattice)
";

%feature("docstring")  Crystal::~Crystal "Crystal::~Crystal()
";

%feature("docstring")  Crystal::clone "Crystal * Crystal::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  Crystal::cloneInvertB "Crystal * Crystal::cloneInvertB() const final

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  Crystal::accept "void Crystal::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  Crystal::setAmbientMaterial "void Crystal::setAmbientMaterial(const IMaterial &material) final
";

%feature("docstring")  Crystal::getAmbientMaterial "const IMaterial * Crystal::getAmbientMaterial() const final

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  Crystal::createTotalFormFactor "IFormFactor * Crystal::createTotalFormFactor(const IFormFactor &meso_crystal_form_factor, const IRotation *p_rotation, const kvector_t &translation) const

Creates a total form factor for the mesocrystal with a specific shape and content The bulk content of the mesocrystal is encapsulated by the  IClusteredParticles object itself 
";

%feature("docstring")  Crystal::getTransformedLattice "Lattice Crystal::getTransformedLattice(const IRotation *p_rotation) const 
";

%feature("docstring")  Crystal::getLatticeBasis "const ParticleComposition* Crystal::getLatticeBasis() const 
";

%feature("docstring")  Crystal::setDWFactor "void Crystal::setDWFactor(double dw_factor)
";

%feature("docstring")  Crystal::getChildren "std::vector< const INode * > Crystal::getChildren() const

Returns a vector of children (const). 
";


// File: classCumulativeValue.xml
%feature("docstring") CumulativeValue "

The cumulative value with average and rms on-the-flight calculations.

C++ includes: CumulativeValue.h
";

%feature("docstring")  CumulativeValue::CumulativeValue "CumulativeValue::CumulativeValue()
";

%feature("docstring")  CumulativeValue::clear "void CumulativeValue::clear()
";

%feature("docstring")  CumulativeValue::setContent "void CumulativeValue::setContent(double value)
";

%feature("docstring")  CumulativeValue::add "void CumulativeValue::add(double value, double weight=1.0)
";

%feature("docstring")  CumulativeValue::getNumberOfEntries "int CumulativeValue::getNumberOfEntries() const 
";

%feature("docstring")  CumulativeValue::getContent "double CumulativeValue::getContent() const 
";

%feature("docstring")  CumulativeValue::getAverage "double CumulativeValue::getAverage() const 
";

%feature("docstring")  CumulativeValue::getRMS "double CumulativeValue::getRMS() const 
";


// File: classCustomBinAxis.xml
%feature("docstring") CustomBinAxis "

Axis with fixed bin size in sin(angle) space used for numerical comparison with IsGisaxs. The main feature of the axis is that it produces zero bin sizes.

C++ includes: CustomBinAxis.h
";

%feature("docstring")  CustomBinAxis::CustomBinAxis "CustomBinAxis::CustomBinAxis(const std::string &name, size_t nbins, double start, double end)

CustomBinAxis constructor.

Parameters:
-----------

name: 
Axis name

nbins: 
number of bins

start: 
center of first bin (IsGisaxs convention)

end: 
center of last bin (IsGisaxs convention) 
";

%feature("docstring")  CustomBinAxis::~CustomBinAxis "virtual CustomBinAxis::~CustomBinAxis()
";

%feature("docstring")  CustomBinAxis::clone "CustomBinAxis * CustomBinAxis::clone() const

clone function 
";

%feature("docstring")  CustomBinAxis::getBin "Bin1D CustomBinAxis::getBin(size_t index) const

retrieve a 1d bin for the given index 
";

%feature("docstring")  CustomBinAxis::getBinCenters "std::vector< double > CustomBinAxis::getBinCenters() const 
";

%feature("docstring")  CustomBinAxis::createClippedAxis "CustomBinAxis * CustomBinAxis::createClippedAxis(double left, double right) const

Creates a new clipped axis. 
";


// File: classCustomMorphologyBuilder.xml
%feature("docstring") CustomMorphologyBuilder "

Builds sample: mixture of different particles (IsGISAXS example #7).

C++ includes: CustomMorphologyBuilder.h
";

%feature("docstring")  CustomMorphologyBuilder::CustomMorphologyBuilder "CustomMorphologyBuilder::CustomMorphologyBuilder()
";

%feature("docstring")  CustomMorphologyBuilder::buildSample "MultiLayer * CustomMorphologyBuilder::buildSample() const 
";


// File: classCylindersAndPrismsBuilder.xml
%feature("docstring") CylindersAndPrismsBuilder "

Builds sample: mixture of cylinders and prisms without interference (IsGISAXS example #1).

C++ includes: CylindersAndPrismsBuilder.h
";

%feature("docstring")  CylindersAndPrismsBuilder::CylindersAndPrismsBuilder "CylindersAndPrismsBuilder::CylindersAndPrismsBuilder()
";

%feature("docstring")  CylindersAndPrismsBuilder::buildSample "MultiLayer * CylindersAndPrismsBuilder::buildSample() const 
";


// File: classCylindersInBABuilder.xml
%feature("docstring") CylindersInBABuilder "

Builds sample: cylinder formfactor in BA (IsGISAXS example #3, part II).

C++ includes: CylindersBuilder.h
";

%feature("docstring")  CylindersInBABuilder::CylindersInBABuilder "CylindersInBABuilder::CylindersInBABuilder()
";

%feature("docstring")  CylindersInBABuilder::buildSample "MultiLayer * CylindersInBABuilder::buildSample() const 
";


// File: classCylindersInDWBABuilder.xml
%feature("docstring") CylindersInDWBABuilder "

Builds sample: cylinder formfactor in DWBA (IsGISAXS example #3, part I).

C++ includes: CylindersBuilder.h
";

%feature("docstring")  CylindersInDWBABuilder::CylindersInDWBABuilder "CylindersInDWBABuilder::CylindersInDWBABuilder()
";

%feature("docstring")  CylindersInDWBABuilder::buildSample "MultiLayer * CylindersInDWBABuilder::buildSample() const 
";


// File: classCylindersInSSCABuilder.xml
%feature("docstring") CylindersInSSCABuilder "

Builds sample: size spacing correlation approximation (IsGISAXS example #15).

C++ includes: SizeDistributionModelsBuilder.h
";

%feature("docstring")  CylindersInSSCABuilder::CylindersInSSCABuilder "CylindersInSSCABuilder::CylindersInSSCABuilder()
";

%feature("docstring")  CylindersInSSCABuilder::buildSample "MultiLayer * CylindersInSSCABuilder::buildSample() const 
";


// File: classCylindersWithSizeDistributionBuilder.xml
%feature("docstring") CylindersWithSizeDistributionBuilder "

Cylinders in BA with size distributions (IsGISAXS example #3, part II).

C++ includes: ParticleDistributionsBuilder.h
";

%feature("docstring")  CylindersWithSizeDistributionBuilder::CylindersWithSizeDistributionBuilder "CylindersWithSizeDistributionBuilder::CylindersWithSizeDistributionBuilder()
";

%feature("docstring")  CylindersWithSizeDistributionBuilder::buildSample "MultiLayer * CylindersWithSizeDistributionBuilder::buildSample() const 
";


// File: classExceptions_1_1DeadReferenceException.xml
%feature("docstring") Exceptions::DeadReferenceException "";

%feature("docstring")  Exceptions::DeadReferenceException::DeadReferenceException "Exceptions::DeadReferenceException::DeadReferenceException(const std::string &message)
";


// File: classDecouplingApproximationStrategy1.xml
%feature("docstring") DecouplingApproximationStrategy1 "

Strategy class to compute the total scalar scattering from a decorated layer in decoupling approximation.

C++ includes: DecouplingApproximationStrategy.h
";

%feature("docstring")  DecouplingApproximationStrategy1::DecouplingApproximationStrategy1 "DecouplingApproximationStrategy1::DecouplingApproximationStrategy1(SimulationOptions sim_params)
";


// File: classDecouplingApproximationStrategy2.xml
%feature("docstring") DecouplingApproximationStrategy2 "

Strategy class to compute the total polarized scattering from a decorated layer in decoupling approximation.

C++ includes: DecouplingApproximationStrategy.h
";

%feature("docstring")  DecouplingApproximationStrategy2::DecouplingApproximationStrategy2 "DecouplingApproximationStrategy2::DecouplingApproximationStrategy2(SimulationOptions sim_params)
";


// File: classDelayedProgressCounter.xml
%feature("docstring") DelayedProgressCounter "

Counter for reporting progress (with delay interval) in a threaded computation.

C++ includes: DelayedProgressCounter.h
";

%feature("docstring")  DelayedProgressCounter::DelayedProgressCounter "DelayedProgressCounter::DelayedProgressCounter(size_t interval)
";

%feature("docstring")  DelayedProgressCounter::~DelayedProgressCounter "DelayedProgressCounter::~DelayedProgressCounter()
";

%feature("docstring")  DelayedProgressCounter::stepProgress "void DelayedProgressCounter::stepProgress(ProgressHandler *progress)

Increments inner counter; at regular intervals updates progress handler. 
";


// File: classDetectionProperties.xml
%feature("docstring") DetectionProperties "

Detector properties (efficiency, transmission).

C++ includes: DetectionProperties.h
";

%feature("docstring")  DetectionProperties::DetectionProperties "DetectionProperties::DetectionProperties()
";

%feature("docstring")  DetectionProperties::setAnalyzerProperties "void DetectionProperties::setAnalyzerProperties(const kvector_t direction, double efficiency, double total_transmission)

Sets the polarization analyzer characteristics of the detector. 
";

%feature("docstring")  DetectionProperties::analyzerOperator "Eigen::Matrix2cd DetectionProperties::analyzerOperator() const

Gets the polarization density matrix (in spin basis along z-axis) 
";


// File: classDetectorMask.xml
%feature("docstring") DetectorMask "

Collection of detector masks.

C++ includes: DetectorMask.h
";

%feature("docstring")  DetectorMask::DetectorMask "DetectorMask::DetectorMask()
";

%feature("docstring")  DetectorMask::DetectorMask "DetectorMask::DetectorMask(const DetectorMask &other)
";

%feature("docstring")  DetectorMask::addMask "void DetectorMask::addMask(const IShape2D &shape, bool mask_value)

Add mask to the stack of detector masks. The value \"true\" means that the area will be excluded from the analysis.

Parameters:
-----------

shape: 
The shape of mask.

mask_value: 
The value of mask 
";

%feature("docstring")  DetectorMask::initMaskData "void DetectorMask::initMaskData(const IDetector2D &detector)

Init the map of masks for the given detector plane. 
";

%feature("docstring")  DetectorMask::initMaskData "void DetectorMask::initMaskData(const OutputData< double > &data)
";

%feature("docstring")  DetectorMask::isMasked "bool DetectorMask::isMasked(size_t index) const 
";

%feature("docstring")  DetectorMask::getMaskData "const OutputData<bool>* DetectorMask::getMaskData() const 
";

%feature("docstring")  DetectorMask::createHistogram "Histogram2D * DetectorMask::createHistogram() const 
";

%feature("docstring")  DetectorMask::removeMasks "void DetectorMask::removeMasks()

remove all masks and return object to initial state 
";

%feature("docstring")  DetectorMask::hasMasks "bool DetectorMask::hasMasks() const

returns true if has masks 
";

%feature("docstring")  DetectorMask::numberOfMaskedChannels "int DetectorMask::numberOfMaskedChannels() const 
";

%feature("docstring")  DetectorMask::numberOfMasks "size_t DetectorMask::numberOfMasks() const 
";

%feature("docstring")  DetectorMask::getMaskShape "const IShape2D * DetectorMask::getMaskShape(size_t mask_index, bool &mask_value) const 
";


// File: classDistributionCosine.xml
%feature("docstring") DistributionCosine "

Cosine distribution.

C++ includes: Distributions.h
";

%feature("docstring")  DistributionCosine::DistributionCosine "DistributionCosine::DistributionCosine()
";

%feature("docstring")  DistributionCosine::DistributionCosine "DistributionCosine::DistributionCosine(double mean, double sigma)
";

%feature("docstring")  DistributionCosine::~DistributionCosine "virtual DistributionCosine::~DistributionCosine()
";

%feature("docstring")  DistributionCosine::clone "DistributionCosine* DistributionCosine::clone() const final
";

%feature("docstring")  DistributionCosine::probabilityDensity "double DistributionCosine::probabilityDensity(double x) const final

Returns the distribution-specific probability density for value x. 
";

%feature("docstring")  DistributionCosine::getMean "double DistributionCosine::getMean() const final

Returns the distribution-specific mean. 
";

%feature("docstring")  DistributionCosine::getSigma "double DistributionCosine::getSigma() const 
";

%feature("docstring")  DistributionCosine::equidistantPoints "std::vector< double > DistributionCosine::equidistantPoints(size_t nbr_samples, double sigma_factor, const RealLimits &limits=RealLimits()) const

generate list of sample values 
";

%feature("docstring")  DistributionCosine::isDelta "bool DistributionCosine::isDelta() const final

Returns true if the distribution is in the limit case of a Dirac delta distribution. 
";


// File: classDistributionGate.xml
%feature("docstring") DistributionGate "

Uniform distribution function with half width hwhm.

C++ includes: Distributions.h
";

%feature("docstring")  DistributionGate::DistributionGate "DistributionGate::DistributionGate()
";

%feature("docstring")  DistributionGate::DistributionGate "DistributionGate::DistributionGate(double min, double max)
";

%feature("docstring")  DistributionGate::~DistributionGate "virtual DistributionGate::~DistributionGate()
";

%feature("docstring")  DistributionGate::clone "DistributionGate* DistributionGate::clone() const final
";

%feature("docstring")  DistributionGate::probabilityDensity "double DistributionGate::probabilityDensity(double x) const final

Returns the distribution-specific probability density for value x. 
";

%feature("docstring")  DistributionGate::getMean "double DistributionGate::getMean() const final

Returns the distribution-specific mean. 
";

%feature("docstring")  DistributionGate::getMin "double DistributionGate::getMin() const 
";

%feature("docstring")  DistributionGate::getMax "double DistributionGate::getMax() const 
";

%feature("docstring")  DistributionGate::equidistantPoints "std::vector< double > DistributionGate::equidistantPoints(size_t nbr_samples, double sigma_factor, const RealLimits &limits=RealLimits()) const

Returns list of sample values. 
";

%feature("docstring")  DistributionGate::isDelta "bool DistributionGate::isDelta() const final

Returns true if the distribution is in the limit case of a Dirac delta distribution. 
";


// File: classDistributionGaussian.xml
%feature("docstring") DistributionGaussian "

Gaussian distribution with standard deviation std_dev.

C++ includes: Distributions.h
";

%feature("docstring")  DistributionGaussian::DistributionGaussian "DistributionGaussian::DistributionGaussian()
";

%feature("docstring")  DistributionGaussian::DistributionGaussian "DistributionGaussian::DistributionGaussian(double mean, double std_dev)
";

%feature("docstring")  DistributionGaussian::~DistributionGaussian "virtual DistributionGaussian::~DistributionGaussian()
";

%feature("docstring")  DistributionGaussian::clone "DistributionGaussian* DistributionGaussian::clone() const final
";

%feature("docstring")  DistributionGaussian::probabilityDensity "double DistributionGaussian::probabilityDensity(double x) const final

Returns the distribution-specific probability density for value x. 
";

%feature("docstring")  DistributionGaussian::getMean "double DistributionGaussian::getMean() const final

Returns the distribution-specific mean. 
";

%feature("docstring")  DistributionGaussian::getStdDev "double DistributionGaussian::getStdDev() const 
";

%feature("docstring")  DistributionGaussian::equidistantPoints "std::vector< double > DistributionGaussian::equidistantPoints(size_t nbr_samples, double sigma_factor, const RealLimits &limits=RealLimits()) const

generate list of sample values 
";

%feature("docstring")  DistributionGaussian::isDelta "bool DistributionGaussian::isDelta() const final

Returns true if the distribution is in the limit case of a Dirac delta distribution. 
";


// File: classDistributionHandler.xml
%feature("docstring") DistributionHandler "

Provides the functionality to average over parameter distributions with weights.

C++ includes: DistributionHandler.h
";

%feature("docstring")  DistributionHandler::DistributionHandler "DistributionHandler::DistributionHandler()
";

%feature("docstring")  DistributionHandler::~DistributionHandler "DistributionHandler::~DistributionHandler()
";

%feature("docstring")  DistributionHandler::addParameterDistribution "void DistributionHandler::addParameterDistribution(const std::string &param_name, const IDistribution1D &distribution, size_t nbr_samples, double sigma_factor=0.0, const RealLimits &limits=RealLimits())

add a sampled parameter distribution 
";

%feature("docstring")  DistributionHandler::addParameterDistribution "void DistributionHandler::addParameterDistribution(const ParameterDistribution &par_distr)
";

%feature("docstring")  DistributionHandler::getTotalNumberOfSamples "size_t DistributionHandler::getTotalNumberOfSamples() const

get the total number of parameter value combinations (product of the individual sizes of each parameter distribution 
";

%feature("docstring")  DistributionHandler::setParameterValues "double DistributionHandler::setParameterValues(ParameterPool *p_parameter_pool, size_t index)

set the parameter values of the simulation object to a specific combination of values, determined by the index (which must be smaller than the total number of combinations) and returns the weight associated with this combination of parameter values 
";

%feature("docstring")  DistributionHandler::getDistributions "const DistributionHandler::Distributions_t & DistributionHandler::getDistributions() const 
";


// File: classDistributionLogNormal.xml
%feature("docstring") DistributionLogNormal "

Log-normal distribution.

C++ includes: Distributions.h
";

%feature("docstring")  DistributionLogNormal::DistributionLogNormal "DistributionLogNormal::DistributionLogNormal(double scale_param)
";

%feature("docstring")  DistributionLogNormal::DistributionLogNormal "DistributionLogNormal::DistributionLogNormal(double median, double scale_param)
";

%feature("docstring")  DistributionLogNormal::~DistributionLogNormal "virtual DistributionLogNormal::~DistributionLogNormal()
";

%feature("docstring")  DistributionLogNormal::clone "DistributionLogNormal* DistributionLogNormal::clone() const final
";

%feature("docstring")  DistributionLogNormal::probabilityDensity "double DistributionLogNormal::probabilityDensity(double x) const final

Returns the distribution-specific probability density for value x. 
";

%feature("docstring")  DistributionLogNormal::getMean "double DistributionLogNormal::getMean() const final

Returns the distribution-specific mean. 
";

%feature("docstring")  DistributionLogNormal::getMedian "double DistributionLogNormal::getMedian() const 
";

%feature("docstring")  DistributionLogNormal::getScalePar "double DistributionLogNormal::getScalePar() const 
";

%feature("docstring")  DistributionLogNormal::equidistantPoints "std::vector< double > DistributionLogNormal::equidistantPoints(size_t nbr_samples, double sigma_factor, const RealLimits &limits=RealLimits()) const

generate list of sample values 
";

%feature("docstring")  DistributionLogNormal::isDelta "bool DistributionLogNormal::isDelta() const final

Returns true if the distribution is in the limit case of a Dirac delta distribution. 
";


// File: classDistributionLorentz.xml
%feature("docstring") DistributionLorentz "

Lorentz distribution with half width hwhm.

C++ includes: Distributions.h
";

%feature("docstring")  DistributionLorentz::DistributionLorentz "DistributionLorentz::DistributionLorentz()
";

%feature("docstring")  DistributionLorentz::DistributionLorentz "DistributionLorentz::DistributionLorentz(double mean, double hwhm)
";

%feature("docstring")  DistributionLorentz::~DistributionLorentz "virtual DistributionLorentz::~DistributionLorentz()
";

%feature("docstring")  DistributionLorentz::clone "DistributionLorentz* DistributionLorentz::clone() const final
";

%feature("docstring")  DistributionLorentz::probabilityDensity "double DistributionLorentz::probabilityDensity(double x) const final

Returns the distribution-specific probability density for value x. 
";

%feature("docstring")  DistributionLorentz::getMean "double DistributionLorentz::getMean() const final

Returns the distribution-specific mean. 
";

%feature("docstring")  DistributionLorentz::getHWHM "double DistributionLorentz::getHWHM() const 
";

%feature("docstring")  DistributionLorentz::equidistantPoints "std::vector< double > DistributionLorentz::equidistantPoints(size_t nbr_samples, double sigma_factor, const RealLimits &limits=RealLimits()) const

generate list of sample values 
";

%feature("docstring")  DistributionLorentz::isDelta "bool DistributionLorentz::isDelta() const final

Returns true if the distribution is in the limit case of a Dirac delta distribution. 
";


// File: classExceptions_1_1DivisionByZeroException.xml
%feature("docstring") Exceptions::DivisionByZeroException "";

%feature("docstring")  Exceptions::DivisionByZeroException::DivisionByZeroException "Exceptions::DivisionByZeroException::DivisionByZeroException(const std::string &message)
";


// File: classExceptions_1_1DomainErrorException.xml
%feature("docstring") Exceptions::DomainErrorException "";

%feature("docstring")  Exceptions::DomainErrorException::DomainErrorException "Exceptions::DomainErrorException::DomainErrorException(const std::string &message)
";


// File: classDWBADiffuseReflection.xml
%feature("docstring") DWBADiffuseReflection "

Calculation of diffuse (aka off-specular) reflection from multilayers.

C++ includes: DWBADiffuseReflection.h
";

%feature("docstring")  DWBADiffuseReflection::DWBADiffuseReflection "DWBADiffuseReflection::DWBADiffuseReflection()
";

%feature("docstring")  DWBADiffuseReflection::execute "void DWBADiffuseReflection::execute(const MultiLayer &sample, const kvector_t ki, const kvector_t kf)
";

%feature("docstring")  DWBADiffuseReflection::setSample "void DWBADiffuseReflection::setSample(const MultiLayer &sample)
";

%feature("docstring")  DWBADiffuseReflection::getDiffuseAutocorr "double DWBADiffuseReflection::getDiffuseAutocorr() const 
";

%feature("docstring")  DWBADiffuseReflection::getDiffuseCrosscorr "double DWBADiffuseReflection::getDiffuseCrosscorr() const 
";

%feature("docstring")  DWBADiffuseReflection::setKvectors "void DWBADiffuseReflection::setKvectors(const kvector_t ki, const kvector_t kf)
";


// File: classEllipse.xml
%feature("docstring") Ellipse "

Ellipse shape.

C++ includes: Ellipse.h
";

%feature("docstring")  Ellipse::Ellipse "Ellipse::Ellipse(double xcenter, double ycenter, double xradius, double yradius, double theta=0.0)

Parameters:
-----------

xcenter: 
x-coordinate of  Ellipse's center

ycenter: 
y-coordinate of  Ellipse's center

xradius: 
Radius along x-axis

yradius: 
Radius along y-axis

theta: 
Angle of  Ellipse rotation in radians 
";

%feature("docstring")  Ellipse::clone "Ellipse* Ellipse::clone() const 
";

%feature("docstring")  Ellipse::contains "bool Ellipse::contains(double x, double y) const

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  Ellipse::contains "bool Ellipse::contains(const Bin1D &binx, const Bin1D &biny) const

Returns true if area defined by two bins is inside or on border of ellipse; more precisely, if mid point of two bins satisfy this condition. 
";

%feature("docstring")  Ellipse::getCenterX "double Ellipse::getCenterX() const 
";

%feature("docstring")  Ellipse::getCenterY "double Ellipse::getCenterY() const 
";

%feature("docstring")  Ellipse::getRadiusX "double Ellipse::getRadiusX() const 
";

%feature("docstring")  Ellipse::getRadiusY "double Ellipse::getRadiusY() const 
";

%feature("docstring")  Ellipse::getTheta "double Ellipse::getTheta() const 
";


// File: classExceptions_1_1ExistingClassRegistrationException.xml
%feature("docstring") Exceptions::ExistingClassRegistrationException "";

%feature("docstring")  Exceptions::ExistingClassRegistrationException::ExistingClassRegistrationException "Exceptions::ExistingClassRegistrationException::ExistingClassRegistrationException(const std::string &message)
";


// File: classExportToPython.xml
%feature("docstring") ExportToPython "

Write a Python script that allows to run the current simulation.

C++ includes: ExportToPython.h
";

%feature("docstring")  ExportToPython::ExportToPython "ExportToPython::ExportToPython(const MultiLayer &multilayer)
";

%feature("docstring")  ExportToPython::~ExportToPython "ExportToPython::~ExportToPython()
";

%feature("docstring")  ExportToPython::simulationToPythonLowlevel "std::string ExportToPython::simulationToPythonLowlevel(const GISASSimulation *simulation)

Returns a Python script that sets up a simulation and runs it if invoked as main program. 
";

%feature("docstring")  ExportToPython::defineGetSample "std::string ExportToPython::defineGetSample() const 
";


// File: classExceptions_1_1FileIsBadException.xml
%feature("docstring") Exceptions::FileIsBadException "";

%feature("docstring")  Exceptions::FileIsBadException::FileIsBadException "Exceptions::FileIsBadException::FileIsBadException(const std::string &message)
";


// File: classExceptions_1_1FileNotIsOpenException.xml
%feature("docstring") Exceptions::FileNotIsOpenException "";

%feature("docstring")  Exceptions::FileNotIsOpenException::FileNotIsOpenException "Exceptions::FileNotIsOpenException::FileNotIsOpenException(const std::string &message)
";


// File: classFitElement.xml
%feature("docstring") FitElement "

Measured (\"real\") and simulated scattering intensity value for one detector cell. Used for chi2/residual calculations.

C++ includes: FitElement.h
";

%feature("docstring")  FitElement::FitElement "FitElement::FitElement()
";

%feature("docstring")  FitElement::FitElement "FitElement::FitElement(size_t index, double simul_value, double real_value, double weight=1.0)
";

%feature("docstring")  FitElement::FitElement "FitElement::FitElement(const FitElement &other)
";

%feature("docstring")  FitElement::getIndex "size_t FitElement::getIndex() const 
";

%feature("docstring")  FitElement::getSimulValue "double FitElement::getSimulValue() const 
";

%feature("docstring")  FitElement::getRealValue "double FitElement::getRealValue() const 
";

%feature("docstring")  FitElement::getWeight "double FitElement::getWeight() const 
";

%feature("docstring")  FitElement::getSquaredDifference "double FitElement::getSquaredDifference() const 
";

%feature("docstring")  FitElement::setSquaredDifference "void FitElement::setSquaredDifference(double value)
";

%feature("docstring")  FitElement::getResidual "double FitElement::getResidual() const 
";

%feature("docstring")  FitElement::setResidual "void FitElement::setResidual(double value)
";


// File: classFitObject.xml
%feature("docstring") FitObject "

Holds simulation description and real data to run the fit.

C++ includes: FitObject.h
";

%feature("docstring")  FitObject::FitObject "FitObject::FitObject(const GISASSimulation &simulation, const OutputData< double > &real_data, double weight=1)

FitObject constructor

Parameters:
-----------

simulation: 
The simulation to run

real_data: 
The real data

weight: 
Weight of dataset in chi2 calculations

adjust_detector_to_data: 
Detector axes will be adjusted to real data axes, if true 
";

%feature("docstring")  FitObject::~FitObject "FitObject::~FitObject()
";

%feature("docstring")  FitObject::accept "void FitObject::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FitObject::realData "const OutputData< double > & FitObject::realData() const

Returns real (experimental) data. 
";

%feature("docstring")  FitObject::simulationData "const OutputData< double > & FitObject::simulationData() const

Returns simulated data. 
";

%feature("docstring")  FitObject::chiSquaredMap "const OutputData< double > & FitObject::chiSquaredMap() const

Returns chi2 map. 
";

%feature("docstring")  FitObject::simulation "const GISASSimulation & FitObject::simulation() const

Returns simulation. 
";

%feature("docstring")  FitObject::weight "double FitObject::weight() const

Returns weight of data set in chi2 calculations. 
";

%feature("docstring")  FitObject::numberOfFitElements "size_t FitObject::numberOfFitElements() const

Returns the size of the data. It is equal to the number of non-masked detector channels which will participate in chi2 calculations. 
";

%feature("docstring")  FitObject::prepareFitElements "void FitObject::prepareFitElements(std::vector< FitElement > &fit_elements, double weight, IIntensityNormalizer *normalizer=0)

Runs simulation and put results (the real and simulated intensities) into external vector. Masked channels will be excluded from the vector. 
";

%feature("docstring")  FitObject::transferToChi2Map "void FitObject::transferToChi2Map(std::vector< FitElement >::const_iterator first, std::vector< FitElement >::const_iterator last) const

Updates ChiSquared map from external vector and returns const reference to it. Used from Python in FitSuiteDrawObserver. 
";

%feature("docstring")  FitObject::getChildren "std::vector< const INode * > FitObject::getChildren() const

Returns a vector of children (const). 
";


// File: classFitParameter.xml
%feature("docstring") FitParameter "

Fittable parameter, linked to other parameters from pools.

C++ includes: FitParameter.h
";

%feature("docstring")  FitParameter::FitParameter "FitParameter::FitParameter()
";

%feature("docstring")  FitParameter::FitParameter "FitParameter::FitParameter(const std::string &pattern, double value, const AttLimits &lim=AttLimits::limitless(), double step=0.0)
";

%feature("docstring")  FitParameter::~FitParameter "FitParameter::~FitParameter() final
";

%feature("docstring")  FitParameter::clone "FitParameter * FitParameter::clone() const 
";

%feature("docstring")  FitParameter::setValue "void FitParameter::setValue(double value) final

Sets given value for all bound parameters. 
";

%feature("docstring")  FitParameter::addPattern "FitParameter & FitParameter::addPattern(const std::string &pattern)

Adds pattern to the list for later usage in parameter pool matching. 
";

%feature("docstring")  FitParameter::addParameter "void FitParameter::addParameter(const RealParameter &par)

Adds real parameter to the collection. 
";

%feature("docstring")  FitParameter::addMatchedParameters "void FitParameter::addMatchedParameters(const ParameterPool &pool)

Adds parameters from pool which match given wildcard. 
";

%feature("docstring")  FitParameter::patterns "std::vector< std::string > FitParameter::patterns() const 
";

%feature("docstring")  FitParameter::matchedParameterNames "std::vector< std::string > FitParameter::matchedParameterNames() const

Returns vector of strings with names of all matched parameters. 
";

%feature("docstring")  FitParameter::patternIntersection "std::vector< std::string > FitParameter::patternIntersection(const FitParameter &other) const

Returns vector containing patterns existing in both FitParametersLinked. 
";

%feature("docstring")  FitParameter::isConflicting "bool FitParameter::isConflicting(const FitParameter &other) const

Returns true if two FitParameterLinked are intended to steer same  RealParameter. 
";


// File: classFitStrategyAdjustParameters.xml
%feature("docstring") FitStrategyAdjustParameters "

Strategy which fixes/releases fit parameters and call minimizer.

C++ includes: FitStrategyAdjustParameters.h
";

%feature("docstring")  FitStrategyAdjustParameters::FitStrategyAdjustParameters "FitStrategyAdjustParameters::FitStrategyAdjustParameters(const std::string &name)
";

%feature("docstring")  FitStrategyAdjustParameters::FitStrategyAdjustParameters "FitStrategyAdjustParameters::FitStrategyAdjustParameters()
";

%feature("docstring")  FitStrategyAdjustParameters::~FitStrategyAdjustParameters "virtual FitStrategyAdjustParameters::~FitStrategyAdjustParameters()
";

%feature("docstring")  FitStrategyAdjustParameters::clone "virtual FitStrategyAdjustParameters* FitStrategyAdjustParameters::clone() const 
";

%feature("docstring")  FitStrategyAdjustParameters::execute "void FitStrategyAdjustParameters::execute()

strategy which fixes/releases fit parameters and then call minimizer 
";

%feature("docstring")  FitStrategyAdjustParameters::fix_all "virtual FitStrategyAdjustParameters& FitStrategyAdjustParameters::fix_all()
";

%feature("docstring")  FitStrategyAdjustParameters::release_all "virtual FitStrategyAdjustParameters& FitStrategyAdjustParameters::release_all()
";

%feature("docstring")  FitStrategyAdjustParameters::fix "virtual FitStrategyAdjustParameters& FitStrategyAdjustParameters::fix(std::string parname)
";

%feature("docstring")  FitStrategyAdjustParameters::release "virtual FitStrategyAdjustParameters& FitStrategyAdjustParameters::release(std::string parname)
";

%feature("docstring")  FitStrategyAdjustParameters::setPreserveOriginalValues "virtual void FitStrategyAdjustParameters::setPreserveOriginalValues(bool preserve_values)
";

%feature("docstring")  FitStrategyAdjustParameters::clear "virtual void FitStrategyAdjustParameters::clear()
";


// File: classFitStrategyDefault.xml
%feature("docstring") FitStrategyDefault "

Default fit strategy just let  FitSuite to run it's minimization round.

C++ includes: IFitStrategy.h
";

%feature("docstring")  FitStrategyDefault::FitStrategyDefault "FitStrategyDefault::FitStrategyDefault()
";

%feature("docstring")  FitStrategyDefault::clone "FitStrategyDefault * FitStrategyDefault::clone() const 
";

%feature("docstring")  FitStrategyDefault::execute "void FitStrategyDefault::execute()
";


// File: classFitStrategyFixParameters.xml
%feature("docstring") FitStrategyFixParameters "

Strategy which fixes certain fit parameters leaving other released.

C++ includes: FitStrategyAdjustParameters.h
";

%feature("docstring")  FitStrategyFixParameters::FitStrategyFixParameters "FitStrategyFixParameters::FitStrategyFixParameters()
";

%feature("docstring")  FitStrategyFixParameters::FitStrategyFixParameters "FitStrategyFixParameters::FitStrategyFixParameters(const std::vector< std::string > &pars)
";

%feature("docstring")  FitStrategyFixParameters::~FitStrategyFixParameters "virtual FitStrategyFixParameters::~FitStrategyFixParameters()
";

%feature("docstring")  FitStrategyFixParameters::clone "virtual FitStrategyFixParameters* FitStrategyFixParameters::clone() const 
";

%feature("docstring")  FitStrategyFixParameters::clear "virtual void FitStrategyFixParameters::clear()
";

%feature("docstring")  FitStrategyFixParameters::setFixed "virtual void FitStrategyFixParameters::setFixed(const std::vector< std::string > &pars)
";


// File: classFitStrategyReleaseParameters.xml
%feature("docstring") FitStrategyReleaseParameters "

Strategy which releases certain fit parameters leaving other fixed.

C++ includes: FitStrategyAdjustParameters.h
";

%feature("docstring")  FitStrategyReleaseParameters::FitStrategyReleaseParameters "FitStrategyReleaseParameters::FitStrategyReleaseParameters()
";

%feature("docstring")  FitStrategyReleaseParameters::FitStrategyReleaseParameters "FitStrategyReleaseParameters::FitStrategyReleaseParameters(const std::vector< std::string > &pars)
";

%feature("docstring")  FitStrategyReleaseParameters::~FitStrategyReleaseParameters "virtual FitStrategyReleaseParameters::~FitStrategyReleaseParameters()
";

%feature("docstring")  FitStrategyReleaseParameters::clone "virtual FitStrategyReleaseParameters* FitStrategyReleaseParameters::clone() const 
";

%feature("docstring")  FitStrategyReleaseParameters::clear "virtual void FitStrategyReleaseParameters::clear()
";

%feature("docstring")  FitStrategyReleaseParameters::setReleased "virtual void FitStrategyReleaseParameters::setReleased(const std::vector< std::string > &pars)
";


// File: classFitSuite.xml
%feature("docstring") FitSuite "

User interface class that wraps all fit methods.

C++ includes: FitSuite.h
";

%feature("docstring")  FitSuite::FitSuite "FitSuite::FitSuite()
";

%feature("docstring")  FitSuite::FitSuite "FitSuite::FitSuite(const FitSuite &)=delete
";

%feature("docstring")  FitSuite::~FitSuite "FitSuite::~FitSuite()
";

%feature("docstring")  FitSuite::addSimulationAndRealData "void FitSuite::addSimulationAndRealData(const GISASSimulation &simulation, const OutputData< double > &real_data, double weight=1)

Assigns pair of (simulation, real data) for fitting. More than one pair can be added. 
";

%feature("docstring")  FitSuite::addSimulationAndRealData "void FitSuite::addSimulationAndRealData(const GISASSimulation &simulation, const IHistogram &real_data, double weight=1)

Assigns pair of (simulation, real data) for fitting. More than one pair can be added. 
";

%feature("docstring")  FitSuite::addSimulationAndRealData "void FitSuite::addSimulationAndRealData(const GISASSimulation &simulation, const std::vector< std::vector< double >> &real_data, double weight=1)

Assigns pair of (simulation, real data) for fitting. Numpy array is used to provide intensities. Shape of array (nrows, ncols) should coinside with detector's axes (n_alpha, n_phi). 
";

%feature("docstring")  FitSuite::addFitParameter "FitParameter * FitSuite::addFitParameter(const std::string &name, double value, const AttLimits &limits=AttLimits::limitless(), double step=0.0)

Adds fit parameter

Parameters:
-----------

name: 
The name of sample parameter(s) to fit (may contain wildcards).

value: 
Parameter's starting value

limits: 
Limits attribute

step: 
Initial parameter's step (some minimizers don't use it) 
";

%feature("docstring")  FitSuite::addFitParameter "FitParameter * FitSuite::addFitParameter(const FitParameter &fitPar)

Adds fit parameter

Parameters:
-----------

fitPar: 
Fully constructed fit parameter. 
";

%feature("docstring")  FitSuite::setMinimizer "void FitSuite::setMinimizer(const std::string &minimizer_name, const std::string &algorithm_name=std::string(), const std::string &minimizer_options=std::string())

Sets minimizer with given name and algorithm type

Parameters:
-----------

minimizer_name: 
The name of the minimizer

algorithm_name: 
Optional name of the minimizer's algorithm

minimizer_options: 
Optional string with additional minimizer settings 
";

%feature("docstring")  FitSuite::setChiSquaredModule "void FitSuite::setChiSquaredModule(const IChiSquaredModule &chi2_module)

Replaces default  ChiSquaredModule with new one. 
";

%feature("docstring")  FitSuite::addFitStrategy "void FitSuite::addFitStrategy(const IFitStrategy &strategy)

Adds fit strategy. 
";

%feature("docstring")  FitSuite::setMinimizer "void FitSuite::setMinimizer(IMinimizer *minimizer)

Sets minimizer. 
";

%feature("docstring")  FitSuite::minimizer "const IMinimizer * FitSuite::minimizer() const

Returns minimizer. 
";

%feature("docstring")  FitSuite::initPrint "void FitSuite::initPrint(int print_every_nth)

Initializes printing to standard output during the fitting. Prints also the summary when completed.

Parameters:
-----------

print_every_nth: 
Print every n'th iteration 
";

%feature("docstring")  FitSuite::runFit "void FitSuite::runFit()

main method to run the fitting 
";

%feature("docstring")  FitSuite::numberOfFitObjects "int FitSuite::numberOfFitObjects() const

Returns number of fit objects, where fit object stands for (real, simulated) pair. 
";

%feature("docstring")  FitSuite::getRealData "IHistogram * FitSuite::getRealData(size_t i_item=0) const

returns real data histogram

Parameters:
-----------

i_item: 
The index of fit object 
";

%feature("docstring")  FitSuite::getSimulationData "IHistogram * FitSuite::getSimulationData(size_t i_item=0) const

returns simulated data histogram

Parameters:
-----------

i_item: 
The index of fit object 
";

%feature("docstring")  FitSuite::getChiSquaredMap "IHistogram * FitSuite::getChiSquaredMap(size_t i_item=0) const

returns chi2 histogram calculated for (real, simulated) data pair

Parameters:
-----------

i_item: 
The index of fit object 
";

%feature("docstring")  FitSuite::fitObjects "FitSuiteObjects * FitSuite::fitObjects()

returns  FitObject (pair of simulation/real data) 
";

%feature("docstring")  FitSuite::fitParameters "FitParameterSet * FitSuite::fitParameters()

Returns reference to fit parameters. 
";

%feature("docstring")  FitSuite::fitStrategies "FitSuiteStrategies * FitSuite::fitStrategies()

Returns reference to fit parameters. 
";

%feature("docstring")  FitSuite::isLastIteration "bool FitSuite::isLastIteration() const

if the last iteration is done (used by observers to print summary) 
";

%feature("docstring")  FitSuite::numberOfIterations "size_t FitSuite::numberOfIterations() const

Returns current number of minimization function calls. 
";

%feature("docstring")  FitSuite::currentStrategyIndex "size_t FitSuite::currentStrategyIndex() const

Returns the number of current strategy. 
";

%feature("docstring")  FitSuite::printResults "void FitSuite::printResults() const

Prints results of the minimization to the standard output. 
";

%feature("docstring")  FitSuite::reportResults "std::string FitSuite::reportResults() const

Reports results of minimization in the form of multi-line string. 
";

%feature("docstring")  FitSuite::getChi2 "double FitSuite::getChi2() const

Returns minimum chi squared value found. 
";

%feature("docstring")  FitSuite::interruptFitting "void FitSuite::interruptFitting()
";

%feature("docstring")  FitSuite::resetInterrupt "void FitSuite::resetInterrupt()
";

%feature("docstring")  FitSuite::isInterrupted "bool FitSuite::isInterrupted()
";

%feature("docstring")  FitSuite::getRealOutputData "const OutputData< double > * FitSuite::getRealOutputData(size_t i_item=0) const 
";

%feature("docstring")  FitSuite::getSimulationOutputData "const OutputData< double > * FitSuite::getSimulationOutputData(size_t i_item=0) const 
";

%feature("docstring")  FitSuite::getChiSquaredOutputData "const OutputData< double > * FitSuite::getChiSquaredOutputData(size_t i_item=0) const 
";

%feature("docstring")  FitSuite::parametersToString "std::string FitSuite::parametersToString() const

Returns multiline string representing possible fit parameters. 
";

%feature("docstring")  FitSuite::treeToString "std::string FitSuite::treeToString() const

Returns multiline string representing tree structure of fit components. 
";

%feature("docstring")  FitSuite::setupToString "std::string FitSuite::setupToString()

Returns multiline string representing fit setup. 
";


// File: classFitSuiteChiSquaredFunction.xml
%feature("docstring") FitSuiteChiSquaredFunction "

Chi squared fitting function for minimizer.

C++ includes: FitSuiteFunctions.h
";

%feature("docstring")  FitSuiteChiSquaredFunction::FitSuiteChiSquaredFunction "FitSuiteChiSquaredFunction::FitSuiteChiSquaredFunction()
";

%feature("docstring")  FitSuiteChiSquaredFunction::~FitSuiteChiSquaredFunction "virtual FitSuiteChiSquaredFunction::~FitSuiteChiSquaredFunction()
";

%feature("docstring")  FitSuiteChiSquaredFunction::evaluate "double FitSuiteChiSquaredFunction::evaluate(const std::vector< double > &pars)

evaluate method for chi2 value called directly from the minimizer

evaluate chi squared value 
";


// File: classFitSuiteGradientFunction.xml
%feature("docstring") FitSuiteGradientFunction "

Gradient fitting function for minimizer.

C++ includes: FitSuiteFunctions.h
";

%feature("docstring")  FitSuiteGradientFunction::FitSuiteGradientFunction "FitSuiteGradientFunction::FitSuiteGradientFunction()
";

%feature("docstring")  FitSuiteGradientFunction::~FitSuiteGradientFunction "virtual FitSuiteGradientFunction::~FitSuiteGradientFunction()
";

%feature("docstring")  FitSuiteGradientFunction::evaluate "double FitSuiteGradientFunction::evaluate(const std::vector< double > &pars, unsigned int index, std::vector< double > &gradients)

evaluate method for gradients and residuals called directly from the minimizer

evaluate residual and derivative for given data element 
";

%feature("docstring")  FitSuiteGradientFunction::getNCallsTotal "virtual size_t FitSuiteGradientFunction::getNCallsTotal() const 
";

%feature("docstring")  FitSuiteGradientFunction::getNCallsGradient "virtual size_t FitSuiteGradientFunction::getNCallsGradient() const 
";


// File: classFitSuiteImpl.xml
%feature("docstring") FitSuiteImpl "

Fitting kernel for  FitSuite.

C++ includes: FitSuiteImpl.h
";

%feature("docstring")  FitSuiteImpl::FitSuiteImpl "FitSuiteImpl::FitSuiteImpl(const std::function< void()> &notifyObservers)
";

%feature("docstring")  FitSuiteImpl::FitSuiteImpl "FitSuiteImpl::FitSuiteImpl(const FitSuiteImpl &)=delete
";

%feature("docstring")  FitSuiteImpl::~FitSuiteImpl "FitSuiteImpl::~FitSuiteImpl()
";

%feature("docstring")  FitSuiteImpl::clear "void FitSuiteImpl::clear()

Resets most state variables, to get prepared for the next fit.

Clears all data. 
";

%feature("docstring")  FitSuiteImpl::addSimulationAndRealData "FitObject * FitSuiteImpl::addSimulationAndRealData(const GISASSimulation &simulation, const OutputData< double > &real_data, double weight)

Adds pair of (simulation, real data) for consecutive simulation. 
";

%feature("docstring")  FitSuiteImpl::addFitParameter "FitParameter * FitSuiteImpl::addFitParameter(const std::string &pattern, double value, const AttLimits &limits, double step=0.0)

Adds fit parameter.

Adds fit parameter, step is calculated from initial parameter value. 
";

%feature("docstring")  FitSuiteImpl::addFitParameter "FitParameter * FitSuiteImpl::addFitParameter(const FitParameter &fitPar)

Adds fit parameter. 
";

%feature("docstring")  FitSuiteImpl::addFitStrategy "void FitSuiteImpl::addFitStrategy(const IFitStrategy &strategy)

Adds fit strategy. 
";

%feature("docstring")  FitSuiteImpl::setMinimizer "void FitSuiteImpl::setMinimizer(IMinimizer *minimizer)

Sets minimizer. 
";

%feature("docstring")  FitSuiteImpl::runFit "void FitSuiteImpl::runFit()

Runs a fit, which may consist of several minimization rounds. 
";

%feature("docstring")  FitSuiteImpl::minimize "void FitSuiteImpl::minimize()

Runs a single minimization round (called by FitSuiteStrategy) 
";

%feature("docstring")  FitSuiteImpl::fitObjects "FitSuiteObjects* FitSuiteImpl::fitObjects()

Returns reference to the kit with data. 
";

%feature("docstring")  FitSuiteImpl::fitObjects "const FitSuiteObjects* FitSuiteImpl::fitObjects() const 
";

%feature("docstring")  FitSuiteImpl::fitParameters "FitParameterSet * FitSuiteImpl::fitParameters()

Returns reference to fit parameters. 
";

%feature("docstring")  FitSuiteImpl::fitStrategies "FitSuiteStrategies* FitSuiteImpl::fitStrategies()

Returns reference to fit parameters. 
";

%feature("docstring")  FitSuiteImpl::isLastIteration "bool FitSuiteImpl::isLastIteration() const

Returns true if the last iteration is done (used by observers to print summary) 
";

%feature("docstring")  FitSuiteImpl::numberOfIterations "size_t FitSuiteImpl::numberOfIterations() const

Returns current number of minimization function calls. 
";

%feature("docstring")  FitSuiteImpl::currentStrategyIndex "size_t FitSuiteImpl::currentStrategyIndex() const

Returns the number of current strategy. 
";

%feature("docstring")  FitSuiteImpl::reportResults "std::string FitSuiteImpl::reportResults() const

Reports results of minimization in the form of multi-line string. 
";

%feature("docstring")  FitSuiteImpl::getOptions "FitOptions& FitSuiteImpl::getOptions()

Returns current fit options. 
";

%feature("docstring")  FitSuiteImpl::setOptions "void FitSuiteImpl::setOptions(const FitOptions &fit_options)

Sets fit options. 
";

%feature("docstring")  FitSuiteImpl::notifyObservers "void FitSuiteImpl::notifyObservers()
";

%feature("docstring")  FitSuiteImpl::interruptFitting "void FitSuiteImpl::interruptFitting()
";

%feature("docstring")  FitSuiteImpl::resetInterrupt "void FitSuiteImpl::resetInterrupt()
";

%feature("docstring")  FitSuiteImpl::isInterrupted "bool FitSuiteImpl::isInterrupted() const 
";

%feature("docstring")  FitSuiteImpl::kernel "const FitKernel * FitSuiteImpl::kernel() const 
";

%feature("docstring")  FitSuiteImpl::setupToString "std::string FitSuiteImpl::setupToString()

Returns multiline string representing fit setup. 
";


// File: classFitSuiteObjects.xml
%feature("docstring") FitSuiteObjects "

Holds vector of  FitObject's (simulation and real data) to fit

C++ includes: FitSuiteObjects.h
";

%feature("docstring")  FitSuiteObjects::FitSuiteObjects "FitSuiteObjects::FitSuiteObjects()
";

%feature("docstring")  FitSuiteObjects::~FitSuiteObjects "FitSuiteObjects::~FitSuiteObjects()
";

%feature("docstring")  FitSuiteObjects::accept "void FitSuiteObjects::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FitSuiteObjects::add "FitObject * FitSuiteObjects::add(const GISASSimulation &simulation, const OutputData< double > &real_data, double weight=1.0)

Adds to kit pair of (simulation, real data) for consecutive simulation. 
";

%feature("docstring")  FitSuiteObjects::getSizeOfDataSet "size_t FitSuiteObjects::getSizeOfDataSet() const

Returns total number of data points (number of all non-masked channels in all fit objects)

Returns total number of data points. 
";

%feature("docstring")  FitSuiteObjects::setChiSquaredModule "void FitSuiteObjects::setChiSquaredModule(const IChiSquaredModule &chi2_module)

Replaces default  ChiSquaredModule with new one. 
";

%feature("docstring")  FitSuiteObjects::getRealData "const OutputData< double > & FitSuiteObjects::getRealData(size_t i_item=0) const

Returns real data from corresponding  FitObject

Parameters:
-----------

i_item: 
Index of  FitObject
";

%feature("docstring")  FitSuiteObjects::getSimulationData "const OutputData< double > & FitSuiteObjects::getSimulationData(size_t i_item=0) const

Returns simulated data from corresponding  FitObject

Parameters:
-----------

i_item: 
Index of  FitObject
";

%feature("docstring")  FitSuiteObjects::getChiSquaredMap "const OutputData< double > & FitSuiteObjects::getChiSquaredMap(size_t i_item=0) const

Returns new chi-squared map from corresponding  FitObject

Parameters:
-----------

i_item: 
Index of  FitObject
";

%feature("docstring")  FitSuiteObjects::runSimulations "void FitSuiteObjects::runSimulations()

run all simulation defined in fit pairs

loop through all defined simulations and run them 
";

%feature("docstring")  FitSuiteObjects::getChiSquaredValue "double FitSuiteObjects::getChiSquaredValue() const

Returns chi2 calculated over whole dataset. 
";

%feature("docstring")  FitSuiteObjects::getResidualValue "double FitSuiteObjects::getResidualValue(size_t global_index)

Returns residuals for single data element

Parameters:
-----------

global_index: 
index accross all element in  FitElement vector 
";

%feature("docstring")  FitSuiteObjects::setNfreeParameters "void FitSuiteObjects::setNfreeParameters(int nfree_parameters)
";

%feature("docstring")  FitSuiteObjects::clear "void FitSuiteObjects::clear()

clear all data 
";

%feature("docstring")  FitSuiteObjects::size "size_t FitSuiteObjects::size() const 
";

%feature("docstring")  FitSuiteObjects::begin "iterator FitSuiteObjects::begin()
";

%feature("docstring")  FitSuiteObjects::end "iterator FitSuiteObjects::end()
";

%feature("docstring")  FitSuiteObjects::getChildren "std::vector< const INode * > FitSuiteObjects::getChildren() const

Returns a vector of children (const). 
";


// File: classFitSuitePrintObserver.xml
%feature("docstring") FitSuitePrintObserver "

Prints fit progress at the end of each  FitSuite's iteration.

C++ includes: FitSuitePrintObserver.h
";

%feature("docstring")  FitSuitePrintObserver::FitSuitePrintObserver "FitSuitePrintObserver::FitSuitePrintObserver(int print_every_nth=1)
";

%feature("docstring")  FitSuitePrintObserver::~FitSuitePrintObserver "virtual FitSuitePrintObserver::~FitSuitePrintObserver()
";

%feature("docstring")  FitSuitePrintObserver::update "void FitSuitePrintObserver::update(FitSuite *fit_suite)

Here Onserver will do actuall job when he thinks that it is a right moment. 
";


// File: classFitSuiteStrategies.xml
%feature("docstring") FitSuiteStrategies "

Collection of strategies to fit.

C++ includes: FitSuiteStrategies.h
";

%feature("docstring")  FitSuiteStrategies::FitSuiteStrategies "FitSuiteStrategies::FitSuiteStrategies()
";

%feature("docstring")  FitSuiteStrategies::~FitSuiteStrategies "FitSuiteStrategies::~FitSuiteStrategies()
";

%feature("docstring")  FitSuiteStrategies::FitSuiteStrategies "FitSuiteStrategies::FitSuiteStrategies(const FitSuiteStrategies &other)=delete
";

%feature("docstring")  FitSuiteStrategies::init "void FitSuiteStrategies::init(FitSuiteImpl *fit_suite)
";

%feature("docstring")  FitSuiteStrategies::addStrategy "void FitSuiteStrategies::addStrategy(const IFitStrategy &strategy)
";

%feature("docstring")  FitSuiteStrategies::minimize "void FitSuiteStrategies::minimize()
";

%feature("docstring")  FitSuiteStrategies::currentStrategyIndex "size_t FitSuiteStrategies::currentStrategyIndex() const 
";

%feature("docstring")  FitSuiteStrategies::size "size_t FitSuiteStrategies::size() const 
";

%feature("docstring")  FitSuiteStrategies::clear "void FitSuiteStrategies::clear()
";

%feature("docstring")  FitSuiteStrategies::currentStrategy "IFitStrategy * FitSuiteStrategies::currentStrategy()
";


// File: classFixedBinAxis.xml
%feature("docstring") FixedBinAxis "

Axis with fixed bin size.

C++ includes: FixedBinAxis.h
";

%feature("docstring")  FixedBinAxis::FixedBinAxis "FixedBinAxis::FixedBinAxis(const std::string &name, size_t nbins, double start, double end)

FixedBinAxis constructor.

Parameters:
-----------

name: 
Axis name

nbins: 
number of bins

start: 
low edge of first bin

end: 
upper edge of last bin 
";

%feature("docstring")  FixedBinAxis::~FixedBinAxis "virtual FixedBinAxis::~FixedBinAxis()
";

%feature("docstring")  FixedBinAxis::clone "FixedBinAxis * FixedBinAxis::clone() const

clone function 
";

%feature("docstring")  FixedBinAxis::size "size_t FixedBinAxis::size() const

retrieve the number of bins 
";

%feature("docstring")  FixedBinAxis::getBin "Bin1D FixedBinAxis::getBin(size_t index) const

retrieve a 1d bin for the given index 
";

%feature("docstring")  FixedBinAxis::getMin "double FixedBinAxis::getMin() const

Returns value of first point of axis. 
";

%feature("docstring")  FixedBinAxis::getMax "double FixedBinAxis::getMax() const

Returns value of last point of axis. 
";

%feature("docstring")  FixedBinAxis::getBinCenter "double FixedBinAxis::getBinCenter(size_t index) const 
";

%feature("docstring")  FixedBinAxis::findClosestIndex "size_t FixedBinAxis::findClosestIndex(double value) const

find bin index which is best match for given value 
";

%feature("docstring")  FixedBinAxis::getBinCenters "std::vector< double > FixedBinAxis::getBinCenters() const 
";

%feature("docstring")  FixedBinAxis::getBinBoundaries "std::vector< double > FixedBinAxis::getBinBoundaries() const 
";

%feature("docstring")  FixedBinAxis::createClippedAxis "FixedBinAxis * FixedBinAxis::createClippedAxis(double left, double right) const

Creates a new clipped axis. 
";


// File: classExceptions_1_1FormatErrorException.xml
%feature("docstring") Exceptions::FormatErrorException "";

%feature("docstring")  Exceptions::FormatErrorException::FormatErrorException "Exceptions::FormatErrorException::FormatErrorException(const std::string &message)
";


// File: classFormFactorAnisoPyramid.xml
%feature("docstring") FormFactorAnisoPyramid "

A frustum (truncated pyramid) with rectangular base.

C++ includes: FormFactorAnisoPyramid.h
";

%feature("docstring")  FormFactorAnisoPyramid::FormFactorAnisoPyramid "FormFactorAnisoPyramid::FormFactorAnisoPyramid(double length, double width, double height, double alpha)

Parameters:
-----------

length: 
of one side of the rectangular base

width: 
of other side of the rectangular base

height: 
of frustum

alpha: 
dihedral angle in radians between base and facet 
";

%feature("docstring")  FormFactorAnisoPyramid::clone "FormFactorAnisoPyramid* FormFactorAnisoPyramid::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorAnisoPyramid::accept "void FormFactorAnisoPyramid::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorAnisoPyramid::getLength "double FormFactorAnisoPyramid::getLength() const 
";

%feature("docstring")  FormFactorAnisoPyramid::getWidth "double FormFactorAnisoPyramid::getWidth() const 
";

%feature("docstring")  FormFactorAnisoPyramid::getHeight "double FormFactorAnisoPyramid::getHeight() const 
";

%feature("docstring")  FormFactorAnisoPyramid::getAlpha "double FormFactorAnisoPyramid::getAlpha() const 
";


// File: classFormFactorBox.xml
%feature("docstring") FormFactorBox "

A rectangular prism (parallelepiped).

C++ includes: FormFactorBox.h
";

%feature("docstring")  FormFactorBox::FormFactorBox "FormFactorBox::FormFactorBox(double length, double width, double height)

Parameters:
-----------

length: 
of rectangular base

width: 
of rectangular base

height: 
of prism 
";

%feature("docstring")  FormFactorBox::clone "FormFactorBox* FormFactorBox::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorBox::accept "void FormFactorBox::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorBox::getLength "double FormFactorBox::getLength() const 
";

%feature("docstring")  FormFactorBox::getHeight "double FormFactorBox::getHeight() const 
";

%feature("docstring")  FormFactorBox::getWidth "double FormFactorBox::getWidth() const 
";

%feature("docstring")  FormFactorBox::getRadialExtension "double FormFactorBox::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorBox::evaluate_for_q "complex_t FormFactorBox::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorCoherentPart.xml
%feature("docstring") FormFactorCoherentPart "

Information about single particle form factor and specular info of the embedding layer.

C++ includes: FormFactorCoherentPart.h
";

%feature("docstring")  FormFactorCoherentPart::FormFactorCoherentPart "FormFactorCoherentPart::FormFactorCoherentPart(IFormFactor *p_ff)
";

%feature("docstring")  FormFactorCoherentPart::FormFactorCoherentPart "FormFactorCoherentPart::FormFactorCoherentPart(const FormFactorCoherentPart &other)
";

%feature("docstring")  FormFactorCoherentPart::~FormFactorCoherentPart "FormFactorCoherentPart::~FormFactorCoherentPart()
";

%feature("docstring")  FormFactorCoherentPart::evaluate "complex_t FormFactorCoherentPart::evaluate(const SimulationElement &sim_element) const 
";

%feature("docstring")  FormFactorCoherentPart::evaluatePol "Eigen::Matrix2cd FormFactorCoherentPart::evaluatePol(const SimulationElement &sim_element) const 
";

%feature("docstring")  FormFactorCoherentPart::setSpecularInfo "void FormFactorCoherentPart::setSpecularInfo(const ILayerSpecularInfo &specular_info)
";

%feature("docstring")  FormFactorCoherentPart::radialExtension "double FormFactorCoherentPart::radialExtension() const 
";


// File: classFormFactorCoherentSum.xml
%feature("docstring") FormFactorCoherentSum "

Information about particle form factor and abundance.

C++ includes: FormFactorCoherentSum.h
";

%feature("docstring")  FormFactorCoherentSum::FormFactorCoherentSum "FormFactorCoherentSum::FormFactorCoherentSum(IFormFactor *ff, double abundance)
";

%feature("docstring")  FormFactorCoherentSum::~FormFactorCoherentSum "FormFactorCoherentSum::~FormFactorCoherentSum()
";

%feature("docstring")  FormFactorCoherentSum::clone "FormFactorCoherentSum * FormFactorCoherentSum::clone() const 
";

%feature("docstring")  FormFactorCoherentSum::evaluate "complex_t FormFactorCoherentSum::evaluate(const SimulationElement &sim_element) const 
";

%feature("docstring")  FormFactorCoherentSum::evaluatePol "Eigen::Matrix2cd FormFactorCoherentSum::evaluatePol(const SimulationElement &sim_element) const 
";

%feature("docstring")  FormFactorCoherentSum::setSpecularInfo "void FormFactorCoherentSum::setSpecularInfo(const ILayerSpecularInfo &specular_info)
";

%feature("docstring")  FormFactorCoherentSum::relativeAbundance "double FormFactorCoherentSum::relativeAbundance() const 
";

%feature("docstring")  FormFactorCoherentSum::scaleRelativeAbundance "void FormFactorCoherentSum::scaleRelativeAbundance(double total_abundance)
";

%feature("docstring")  FormFactorCoherentSum::radialExtension "double FormFactorCoherentSum::radialExtension() const 
";


// File: classFormFactorCone.xml
%feature("docstring") FormFactorCone "

A conical frustum (cone truncated parallel to the base) with circular base.

C++ includes: FormFactorCone.h
";

%feature("docstring")  FormFactorCone::FormFactorCone "FormFactorCone::FormFactorCone(double radius, double height, double alpha)

Parameters:
-----------

radius: 
of circular base

height: 
of frustum

alpha: 
angle in radians between base and lateral surface 
";

%feature("docstring")  FormFactorCone::clone "FormFactorCone* FormFactorCone::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorCone::accept "void FormFactorCone::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorCone::getHeight "double FormFactorCone::getHeight() const 
";

%feature("docstring")  FormFactorCone::getAlpha "double FormFactorCone::getAlpha() const 
";

%feature("docstring")  FormFactorCone::getRadius "double FormFactorCone::getRadius() const 
";

%feature("docstring")  FormFactorCone::getRadialExtension "double FormFactorCone::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorCone::evaluate_for_q "complex_t FormFactorCone::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorCone6.xml
%feature("docstring") FormFactorCone6 "

A frustum (truncated pyramid) with regular hexagonal base.

C++ includes: FormFactorCone6.h
";

%feature("docstring")  FormFactorCone6::FormFactorCone6 "FormFactorCone6::FormFactorCone6(double base_edge, double height, double alpha)

Parameters:
-----------

base_edge: 
of hexagonal base

height: 
of frustum

alpha: 
dihedral angle in radians between base and facet 
";

%feature("docstring")  FormFactorCone6::clone "FormFactorCone6* FormFactorCone6::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorCone6::accept "void FormFactorCone6::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorCone6::getBaseEdge "double FormFactorCone6::getBaseEdge() const 
";

%feature("docstring")  FormFactorCone6::getHeight "double FormFactorCone6::getHeight() const 
";

%feature("docstring")  FormFactorCone6::getAlpha "double FormFactorCone6::getAlpha() const 
";


// File: classFormFactorCrystal.xml
%feature("docstring") FormFactorCrystal "

The formfactor of a  MesoCrystal.

C++ includes: FormFactorCrystal.h
";

%feature("docstring")  FormFactorCrystal::FormFactorCrystal "FormFactorCrystal::FormFactorCrystal(const Lattice &lattice, const IFormFactor &basis_form_factor, const IFormFactor &meso_form_factor)
";

%feature("docstring")  FormFactorCrystal::~FormFactorCrystal "FormFactorCrystal::~FormFactorCrystal() overridefinal
";

%feature("docstring")  FormFactorCrystal::clone "FormFactorCrystal* FormFactorCrystal::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorCrystal::accept "void FormFactorCrystal::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorCrystal::getVolume "double FormFactorCrystal::getVolume() const overridefinal

Returns the total volume of the particle of this form factor's shape. 
";

%feature("docstring")  FormFactorCrystal::getRadialExtension "double FormFactorCrystal::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorCrystal::evaluate "complex_t FormFactorCrystal::evaluate(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  FormFactorCrystal::evaluatePol "Eigen::Matrix2cd FormFactorCrystal::evaluatePol(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for matrix interactions. 
";


// File: classFormFactorCuboctahedron.xml
%feature("docstring") FormFactorCuboctahedron "

A truncated bifrustum with quadratic base.

C++ includes: FormFactorCuboctahedron.h
";

%feature("docstring")  FormFactorCuboctahedron::FormFactorCuboctahedron "FormFactorCuboctahedron::FormFactorCuboctahedron(double length, double height, double height_ratio, double alpha)

Parameters:
-----------

length: 
of one side of the square base

height: 
of bottom frustum

height_ratio: 
ratio of heights of top to bottom frustum

alpha: 
dihedral angle in radians between base and facet 
";

%feature("docstring")  FormFactorCuboctahedron::clone "FormFactorCuboctahedron* FormFactorCuboctahedron::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorCuboctahedron::accept "void FormFactorCuboctahedron::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorCuboctahedron::getLength "double FormFactorCuboctahedron::getLength() const 
";

%feature("docstring")  FormFactorCuboctahedron::getHeight "double FormFactorCuboctahedron::getHeight() const 
";

%feature("docstring")  FormFactorCuboctahedron::getHeightRatio "double FormFactorCuboctahedron::getHeightRatio() const 
";

%feature("docstring")  FormFactorCuboctahedron::getAlpha "double FormFactorCuboctahedron::getAlpha() const 
";


// File: classFormFactorCylinder.xml
%feature("docstring") FormFactorCylinder "

A circular cylinder.

C++ includes: FormFactorCylinder.h
";

%feature("docstring")  FormFactorCylinder::FormFactorCylinder "FormFactorCylinder::FormFactorCylinder(double radius, double height)
";

%feature("docstring")  FormFactorCylinder::clone "FormFactorCylinder* FormFactorCylinder::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorCylinder::accept "void FormFactorCylinder::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorCylinder::getHeight "double FormFactorCylinder::getHeight() const 
";

%feature("docstring")  FormFactorCylinder::getRadius "double FormFactorCylinder::getRadius() const 
";

%feature("docstring")  FormFactorCylinder::getRadialExtension "double FormFactorCylinder::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorCylinder::evaluate_for_q "complex_t FormFactorCylinder::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorDecoratorDebyeWaller.xml
%feature("docstring") FormFactorDecoratorDebyeWaller "

Debye-Waller factors in radial and z directions.

C++ includes: FormFactorDecoratorDebyeWaller.h
";

%feature("docstring")  FormFactorDecoratorDebyeWaller::FormFactorDecoratorDebyeWaller "FormFactorDecoratorDebyeWaller::FormFactorDecoratorDebyeWaller(const IFormFactor &form_factor, double dw_h_factor, double dw_r_factor)

Anisotropic Debye-Waller factor. 
";

%feature("docstring")  FormFactorDecoratorDebyeWaller::FormFactorDecoratorDebyeWaller "FormFactorDecoratorDebyeWaller::FormFactorDecoratorDebyeWaller(const IFormFactor &form_factor, double dw_factor)

Isotropic Debye-Waller factor. 
";

%feature("docstring")  FormFactorDecoratorDebyeWaller::clone "FormFactorDecoratorDebyeWaller* FormFactorDecoratorDebyeWaller::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDecoratorDebyeWaller::accept "void FormFactorDecoratorDebyeWaller::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDecoratorDebyeWaller::evaluate "complex_t FormFactorDecoratorDebyeWaller::evaluate(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  FormFactorDecoratorDebyeWaller::evaluatePol "Eigen::Matrix2cd FormFactorDecoratorDebyeWaller::evaluatePol(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for matrix interactions. 
";


// File: classFormFactorDecoratorFactor.xml
%feature("docstring") FormFactorDecoratorFactor "

Decorates a formfactor with a constant factor. Base class of  FormFactorDecoratorMaterial.

C++ includes: FormFactorDecoratorFactor.h
";

%feature("docstring")  FormFactorDecoratorFactor::FormFactorDecoratorFactor "FormFactorDecoratorFactor::FormFactorDecoratorFactor(const IFormFactor &form_factor, const complex_t factor)
";

%feature("docstring")  FormFactorDecoratorFactor::clone "FormFactorDecoratorFactor* FormFactorDecoratorFactor::clone() const override

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDecoratorFactor::accept "void FormFactorDecoratorFactor::accept(INodeVisitor *visitor) const override

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDecoratorFactor::evaluate "complex_t FormFactorDecoratorFactor::evaluate(const WavevectorInfo &wavevectors) const override

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  FormFactorDecoratorFactor::evaluatePol "Eigen::Matrix2cd FormFactorDecoratorFactor::evaluatePol(const WavevectorInfo &wavevectors) const override

Returns scattering amplitude for matrix interactions. 
";


// File: classFormFactorDecoratorMaterial.xml
%feature("docstring") FormFactorDecoratorMaterial "

Decorates a scalar formfactor with the correct factor for the material's refractive index and that of its surrounding material.

C++ includes: FormFactorDecoratorMaterial.h
";

%feature("docstring")  FormFactorDecoratorMaterial::FormFactorDecoratorMaterial "FormFactorDecoratorMaterial::FormFactorDecoratorMaterial(const IFormFactor &form_factor)
";

%feature("docstring")  FormFactorDecoratorMaterial::~FormFactorDecoratorMaterial "FormFactorDecoratorMaterial::~FormFactorDecoratorMaterial() overridefinal
";

%feature("docstring")  FormFactorDecoratorMaterial::clone "FormFactorDecoratorMaterial * FormFactorDecoratorMaterial::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDecoratorMaterial::accept "void FormFactorDecoratorMaterial::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDecoratorMaterial::setMaterial "void FormFactorDecoratorMaterial::setMaterial(const IMaterial &material)

Sets the material of the scatterer. 
";

%feature("docstring")  FormFactorDecoratorMaterial::setAmbientMaterial "void FormFactorDecoratorMaterial::setAmbientMaterial(const IMaterial &material) override

Sets the ambient material. 
";

%feature("docstring")  FormFactorDecoratorMaterial::getAmbientRefractiveIndex "complex_t FormFactorDecoratorMaterial::getAmbientRefractiveIndex() const 
";

%feature("docstring")  FormFactorDecoratorMaterial::evaluate "complex_t FormFactorDecoratorMaterial::evaluate(const WavevectorInfo &wavevectors) const override

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  FormFactorDecoratorMaterial::evaluatePol "Eigen::Matrix2cd FormFactorDecoratorMaterial::evaluatePol(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for matrix interactions. 
";


// File: classFormFactorDecoratorPositionFactor.xml
%feature("docstring") FormFactorDecoratorPositionFactor "

Decorates a formfactor with a position dependent phase factor.

C++ includes: FormFactorDecoratorPositionFactor.h
";

%feature("docstring")  FormFactorDecoratorPositionFactor::FormFactorDecoratorPositionFactor "FormFactorDecoratorPositionFactor::FormFactorDecoratorPositionFactor(const IFormFactor &form_factor, const kvector_t &position)
";

%feature("docstring")  FormFactorDecoratorPositionFactor::clone "FormFactorDecoratorPositionFactor* FormFactorDecoratorPositionFactor::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDecoratorPositionFactor::accept "void FormFactorDecoratorPositionFactor::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDecoratorPositionFactor::evaluate "complex_t FormFactorDecoratorPositionFactor::evaluate(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  FormFactorDecoratorPositionFactor::evaluatePol "Eigen::Matrix2cd FormFactorDecoratorPositionFactor::evaluatePol(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for matrix interactions. 
";


// File: classFormFactorDecoratorRotation.xml
%feature("docstring") FormFactorDecoratorRotation "

Equips a formfactor with a rotation.

C++ includes: FormFactorDecoratorRotation.h
";

%feature("docstring")  FormFactorDecoratorRotation::FormFactorDecoratorRotation "FormFactorDecoratorRotation::FormFactorDecoratorRotation(const IFormFactor &form_factor, const IRotation &rotation)

Constructor, setting form factor and rotation. 
";

%feature("docstring")  FormFactorDecoratorRotation::clone "FormFactorDecoratorRotation * FormFactorDecoratorRotation::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDecoratorRotation::accept "void FormFactorDecoratorRotation::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDecoratorRotation::evaluate "complex_t FormFactorDecoratorRotation::evaluate(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  FormFactorDecoratorRotation::evaluatePol "Eigen::Matrix2cd FormFactorDecoratorRotation::evaluatePol(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for matrix interactions. 
";


// File: classFormFactorDodecahedron.xml
%feature("docstring") FormFactorDodecahedron "

A regular dodecahedron.

C++ includes: FormFactorDodecahedron.h
";

%feature("docstring")  FormFactorDodecahedron::FormFactorDodecahedron "FormFactorDodecahedron::FormFactorDodecahedron(double edge)

Constructs a regular dodecahedron.

Parameters:
-----------

edge: 
length 
";

%feature("docstring")  FormFactorDodecahedron::clone "FormFactorDodecahedron* FormFactorDodecahedron::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDodecahedron::accept "void FormFactorDodecahedron::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDodecahedron::getEdge "double FormFactorDodecahedron::getEdge() const 
";


// File: classFormFactorDWBA.xml
%feature("docstring") FormFactorDWBA "

Evaluates the coherent sum of the four DWBA terms in a scalar  IFormFactor.

C++ includes: FormFactorDWBA.h
";

%feature("docstring")  FormFactorDWBA::FormFactorDWBA "FormFactorDWBA::FormFactorDWBA(const IFormFactor &form_factor)
";

%feature("docstring")  FormFactorDWBA::~FormFactorDWBA "FormFactorDWBA::~FormFactorDWBA() override
";

%feature("docstring")  FormFactorDWBA::clone "FormFactorDWBA * FormFactorDWBA::clone() const override

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDWBA::accept "void FormFactorDWBA::accept(INodeVisitor *visitor) const override

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDWBA::evaluate "complex_t FormFactorDWBA::evaluate(const WavevectorInfo &wavevectors) const override

Calculates and returns a form factor calculation in DWBA. 
";

%feature("docstring")  FormFactorDWBA::getVolume "double FormFactorDWBA::getVolume() const override

Returns the total volume of the particle of this form factor's shape. 
";

%feature("docstring")  FormFactorDWBA::getRadialExtension "double FormFactorDWBA::getRadialExtension() const override

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorDWBA::setSpecularInfo "void FormFactorDWBA::setSpecularInfo(const ILayerRTCoefficients *p_in_coeffs, const ILayerRTCoefficients *p_out_coeffs) override

Sets reflection/transmission info for scalar DWBA simulation. 
";


// File: classFormFactorDWBAPol.xml
%feature("docstring") FormFactorDWBAPol "

Evaluates the coherent sum of the 16 matrix DWBA terms in a polarized  IFormFactor.

C++ includes: FormFactorDWBAPol.h
";

%feature("docstring")  FormFactorDWBAPol::FormFactorDWBAPol "FormFactorDWBAPol::FormFactorDWBAPol(const IFormFactor &form_factor)
";

%feature("docstring")  FormFactorDWBAPol::~FormFactorDWBAPol "FormFactorDWBAPol::~FormFactorDWBAPol() override
";

%feature("docstring")  FormFactorDWBAPol::clone "FormFactorDWBAPol * FormFactorDWBAPol::clone() const override

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorDWBAPol::accept "void FormFactorDWBAPol::accept(INodeVisitor *visitor) const override

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorDWBAPol::evaluate "complex_t FormFactorDWBAPol::evaluate(const WavevectorInfo &wavevectors) const override

Throws not-implemented exception. 
";

%feature("docstring")  FormFactorDWBAPol::evaluatePol "Eigen::Matrix2cd FormFactorDWBAPol::evaluatePol(const WavevectorInfo &wavevectors) const override

Calculates and returns a polarized form factor calculation in DWBA. 
";

%feature("docstring")  FormFactorDWBAPol::getVolume "double FormFactorDWBAPol::getVolume() const override

Returns the total volume of the particle of this form factor's shape. 
";

%feature("docstring")  FormFactorDWBAPol::getRadialExtension "double FormFactorDWBAPol::getRadialExtension() const override

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorDWBAPol::setSpecularInfo "void FormFactorDWBAPol::setSpecularInfo(const ILayerRTCoefficients *p_in_coeffs, const ILayerRTCoefficients *p_out_coeffs) override

Sets reflection/transmission info for scalar DWBA simulation. 
";


// File: classFormFactorEllipsoidalCylinder.xml
%feature("docstring") FormFactorEllipsoidalCylinder "

A cylinder with elliptical base.

C++ includes: FormFactorEllipsoidalCylinder.h
";

%feature("docstring")  FormFactorEllipsoidalCylinder::FormFactorEllipsoidalCylinder "FormFactorEllipsoidalCylinder::FormFactorEllipsoidalCylinder(double radius_x, double radius_y, double height)

Parameters:
-----------

radius_x: 
half length of one horizontal main axes

radius_y: 
half length of the other horizontal main axes

height: 
";

%feature("docstring")  FormFactorEllipsoidalCylinder::clone "FormFactorEllipsoidalCylinder* FormFactorEllipsoidalCylinder::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorEllipsoidalCylinder::accept "void FormFactorEllipsoidalCylinder::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorEllipsoidalCylinder::getRadiusX "double FormFactorEllipsoidalCylinder::getRadiusX() const 
";

%feature("docstring")  FormFactorEllipsoidalCylinder::getRadiusY "double FormFactorEllipsoidalCylinder::getRadiusY() const 
";

%feature("docstring")  FormFactorEllipsoidalCylinder::getHeight "double FormFactorEllipsoidalCylinder::getHeight() const 
";

%feature("docstring")  FormFactorEllipsoidalCylinder::getRadialExtension "double FormFactorEllipsoidalCylinder::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorEllipsoidalCylinder::evaluate_for_q "complex_t FormFactorEllipsoidalCylinder::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorFullSphere.xml
%feature("docstring") FormFactorFullSphere "

A full sphere.

C++ includes: FormFactorFullSphere.h
";

%feature("docstring")  FormFactorFullSphere::FormFactorFullSphere "FormFactorFullSphere::FormFactorFullSphere(double radius)
";

%feature("docstring")  FormFactorFullSphere::clone "FormFactorFullSphere* FormFactorFullSphere::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorFullSphere::accept "void FormFactorFullSphere::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorFullSphere::getRadius "double FormFactorFullSphere::getRadius() const 
";

%feature("docstring")  FormFactorFullSphere::getRadialExtension "double FormFactorFullSphere::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorFullSphere::evaluate_for_q "complex_t FormFactorFullSphere::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorFullSpheroid.xml
%feature("docstring") FormFactorFullSpheroid "

A full spheroid (an ellipsoid with two equal axes, hence with circular cross section)

C++ includes: FormFactorFullSpheroid.h
";

%feature("docstring")  FormFactorFullSpheroid::FormFactorFullSpheroid "FormFactorFullSpheroid::FormFactorFullSpheroid(double radius, double height)

Parameters:
-----------

radius: 
of the two equal axes

height: 
total height of the spheroid, i.e. twice the radius of the third axis 
";

%feature("docstring")  FormFactorFullSpheroid::clone "FormFactorFullSpheroid* FormFactorFullSpheroid::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorFullSpheroid::accept "void FormFactorFullSpheroid::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorFullSpheroid::getHeight "double FormFactorFullSpheroid::getHeight() const 
";

%feature("docstring")  FormFactorFullSpheroid::getRadius "double FormFactorFullSpheroid::getRadius() const 
";

%feature("docstring")  FormFactorFullSpheroid::getRadialExtension "double FormFactorFullSpheroid::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorFullSpheroid::evaluate_for_q "complex_t FormFactorFullSpheroid::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorGauss.xml
%feature("docstring") FormFactorGauss "

The formfactor of a gaussian.

C++ includes: FormFactorGauss.h
";

%feature("docstring")  FormFactorGauss::FormFactorGauss "FormFactorGauss::FormFactorGauss(double length)
";

%feature("docstring")  FormFactorGauss::FormFactorGauss "FormFactorGauss::FormFactorGauss(double width, double height)
";

%feature("docstring")  FormFactorGauss::clone "FormFactorGauss* FormFactorGauss::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorGauss::accept "void FormFactorGauss::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorGauss::getWidth "double FormFactorGauss::getWidth() const 
";

%feature("docstring")  FormFactorGauss::getHeight "double FormFactorGauss::getHeight() const 
";

%feature("docstring")  FormFactorGauss::getRadialExtension "double FormFactorGauss::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorGauss::evaluate_for_q "complex_t FormFactorGauss::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorHemiEllipsoid.xml
%feature("docstring") FormFactorHemiEllipsoid "

An hemi ellipsoid, obtained by truncating a full ellipsoid in the middle plane spanned by two principal axes.

C++ includes: FormFactorHemiEllipsoid.h
";

%feature("docstring")  FormFactorHemiEllipsoid::FormFactorHemiEllipsoid "FormFactorHemiEllipsoid::FormFactorHemiEllipsoid(double radius_x, double radius_y, double height)

Parameters:
-----------

radius_x: 
half length of one horizontal main axes

radius_y: 
half length of the other horizontal main axes

height: 
of the hemi ellipsoid 
";

%feature("docstring")  FormFactorHemiEllipsoid::~FormFactorHemiEllipsoid "virtual FormFactorHemiEllipsoid::~FormFactorHemiEllipsoid()
";

%feature("docstring")  FormFactorHemiEllipsoid::clone "FormFactorHemiEllipsoid* FormFactorHemiEllipsoid::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorHemiEllipsoid::accept "void FormFactorHemiEllipsoid::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorHemiEllipsoid::getHeight "double FormFactorHemiEllipsoid::getHeight() const 
";

%feature("docstring")  FormFactorHemiEllipsoid::getRadiusX "double FormFactorHemiEllipsoid::getRadiusX() const 
";

%feature("docstring")  FormFactorHemiEllipsoid::getRadiusY "double FormFactorHemiEllipsoid::getRadiusY() const 
";

%feature("docstring")  FormFactorHemiEllipsoid::getRadialExtension "double FormFactorHemiEllipsoid::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorHemiEllipsoid::evaluate_for_q "complex_t FormFactorHemiEllipsoid::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorIcosahedron.xml
%feature("docstring") FormFactorIcosahedron "

A regular icosahedron.

C++ includes: FormFactorIcosahedron.h
";

%feature("docstring")  FormFactorIcosahedron::FormFactorIcosahedron "FormFactorIcosahedron::FormFactorIcosahedron(double edge)
";

%feature("docstring")  FormFactorIcosahedron::clone "FormFactorIcosahedron* FormFactorIcosahedron::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorIcosahedron::accept "void FormFactorIcosahedron::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorIcosahedron::getEdge "double FormFactorIcosahedron::getEdge() const 
";


// File: classFormFactorLongBoxGauss.xml
%feature("docstring") FormFactorLongBoxGauss "

The formfactor for a long rectangular box.

C++ includes: FormFactorLongBoxGauss.h
";

%feature("docstring")  FormFactorLongBoxGauss::FormFactorLongBoxGauss "FormFactorLongBoxGauss::FormFactorLongBoxGauss(double length, double width, double height)

Box constructor.

Parameters:
-----------

length: 
of Box's base

width: 
of Box's base

height: 
of Box 
";

%feature("docstring")  FormFactorLongBoxGauss::clone "FormFactorLongBoxGauss* FormFactorLongBoxGauss::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorLongBoxGauss::accept "void FormFactorLongBoxGauss::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorLongBoxGauss::getLength "double FormFactorLongBoxGauss::getLength() const 
";

%feature("docstring")  FormFactorLongBoxGauss::getHeight "double FormFactorLongBoxGauss::getHeight() const 
";

%feature("docstring")  FormFactorLongBoxGauss::getWidth "double FormFactorLongBoxGauss::getWidth() const 
";

%feature("docstring")  FormFactorLongBoxGauss::getRadialExtension "double FormFactorLongBoxGauss::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorLongBoxGauss::evaluate_for_q "complex_t FormFactorLongBoxGauss::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorLongBoxLorentz.xml
%feature("docstring") FormFactorLongBoxLorentz "

The formfactor for a long rectangular box.

C++ includes: FormFactorLongBoxLorentz.h
";

%feature("docstring")  FormFactorLongBoxLorentz::FormFactorLongBoxLorentz "FormFactorLongBoxLorentz::FormFactorLongBoxLorentz(double length, double width, double height)

Box constructor.

Parameters:
-----------

length: 
of Box's base

width: 
of Box's base

height: 
of Box 
";

%feature("docstring")  FormFactorLongBoxLorentz::clone "FormFactorLongBoxLorentz* FormFactorLongBoxLorentz::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorLongBoxLorentz::accept "void FormFactorLongBoxLorentz::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorLongBoxLorentz::getLength "double FormFactorLongBoxLorentz::getLength() const 
";

%feature("docstring")  FormFactorLongBoxLorentz::getHeight "double FormFactorLongBoxLorentz::getHeight() const 
";

%feature("docstring")  FormFactorLongBoxLorentz::getWidth "double FormFactorLongBoxLorentz::getWidth() const 
";

%feature("docstring")  FormFactorLongBoxLorentz::getRadialExtension "double FormFactorLongBoxLorentz::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorLongBoxLorentz::evaluate_for_q "complex_t FormFactorLongBoxLorentz::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorLongRipple1Gauss.xml
%feature("docstring") FormFactorLongRipple1Gauss "

The formfactor for a cosine ripple.

C++ includes: FormFactorLongRipple1Gauss.h
";

%feature("docstring")  FormFactorLongRipple1Gauss::FormFactorLongRipple1Gauss "FormFactorLongRipple1Gauss::FormFactorLongRipple1Gauss(double length, double width, double height)

Ripple1 constructor.

Parameters:
-----------

length: 
of Ripple1

width: 
of cosine cross section

height: 
of cosine cross section 
";

%feature("docstring")  FormFactorLongRipple1Gauss::clone "FormFactorLongRipple1Gauss* FormFactorLongRipple1Gauss::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorLongRipple1Gauss::accept "void FormFactorLongRipple1Gauss::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorLongRipple1Gauss::getHeight "double FormFactorLongRipple1Gauss::getHeight() const 
";

%feature("docstring")  FormFactorLongRipple1Gauss::getWidth "double FormFactorLongRipple1Gauss::getWidth() const 
";

%feature("docstring")  FormFactorLongRipple1Gauss::getLength "double FormFactorLongRipple1Gauss::getLength() const 
";

%feature("docstring")  FormFactorLongRipple1Gauss::getRadialExtension "double FormFactorLongRipple1Gauss::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorLongRipple1Gauss::evaluate_for_q "complex_t FormFactorLongRipple1Gauss::evaluate_for_q(const cvector_t q) const overridefinal

Complex formfactor. 
";


// File: classFormFactorLongRipple1Lorentz.xml
%feature("docstring") FormFactorLongRipple1Lorentz "

The formfactor for a cosine ripple.

C++ includes: FormFactorLongRipple1Lorentz.h
";

%feature("docstring")  FormFactorLongRipple1Lorentz::FormFactorLongRipple1Lorentz "FormFactorLongRipple1Lorentz::FormFactorLongRipple1Lorentz(double length, double width, double height)

FormFactorLongRipple1Lorentz constructor.

Parameters:
-----------

length: 
of Ripple1

width: 
of cosine cross section

height: 
of cosine cross section 
";

%feature("docstring")  FormFactorLongRipple1Lorentz::clone "FormFactorLongRipple1Lorentz* FormFactorLongRipple1Lorentz::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorLongRipple1Lorentz::accept "void FormFactorLongRipple1Lorentz::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorLongRipple1Lorentz::getRadialExtension "double FormFactorLongRipple1Lorentz::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorLongRipple1Lorentz::getHeight "double FormFactorLongRipple1Lorentz::getHeight() const 
";

%feature("docstring")  FormFactorLongRipple1Lorentz::getWidth "double FormFactorLongRipple1Lorentz::getWidth() const 
";

%feature("docstring")  FormFactorLongRipple1Lorentz::getLength "double FormFactorLongRipple1Lorentz::getLength() const 
";

%feature("docstring")  FormFactorLongRipple1Lorentz::evaluate_for_q "complex_t FormFactorLongRipple1Lorentz::evaluate_for_q(const cvector_t q) const overridefinal

Complex formfactor. 
";


// File: classFormFactorLongRipple2Gauss.xml
%feature("docstring") FormFactorLongRipple2Gauss "

The formfactor for a triangular ripple.

C++ includes: FormFactorLongRipple2Gauss.h
";

%feature("docstring")  FormFactorLongRipple2Gauss::FormFactorLongRipple2Gauss "FormFactorLongRipple2Gauss::FormFactorLongRipple2Gauss(double length, double width, double height, double asymmetry)

FormFactorLongRipple2Gauss constructor.

Parameters:
-----------

length: 
of Ripple2

width: 
of triangular cross section

height: 
of triangular cross section

asymmetry: 
length of triangular cross section 
";

%feature("docstring")  FormFactorLongRipple2Gauss::clone "FormFactorLongRipple2Gauss* FormFactorLongRipple2Gauss::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorLongRipple2Gauss::accept "void FormFactorLongRipple2Gauss::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorLongRipple2Gauss::getHeight "double FormFactorLongRipple2Gauss::getHeight() const 
";

%feature("docstring")  FormFactorLongRipple2Gauss::getWidth "double FormFactorLongRipple2Gauss::getWidth() const 
";

%feature("docstring")  FormFactorLongRipple2Gauss::getLength "double FormFactorLongRipple2Gauss::getLength() const 
";

%feature("docstring")  FormFactorLongRipple2Gauss::getAsymmetry "double FormFactorLongRipple2Gauss::getAsymmetry() const 
";

%feature("docstring")  FormFactorLongRipple2Gauss::getRadialExtension "double FormFactorLongRipple2Gauss::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorLongRipple2Gauss::evaluate_for_q "complex_t FormFactorLongRipple2Gauss::evaluate_for_q(const cvector_t q) const overridefinal

Complex formfactor. 
";


// File: classFormFactorLongRipple2Lorentz.xml
%feature("docstring") FormFactorLongRipple2Lorentz "

The formfactor for a triangular ripple.

C++ includes: FormFactorLongRipple2Lorentz.h
";

%feature("docstring")  FormFactorLongRipple2Lorentz::FormFactorLongRipple2Lorentz "FormFactorLongRipple2Lorentz::FormFactorLongRipple2Lorentz(double length, double width, double height, double asymmetry)

Parameters:
-----------

length: 
of Ripple2

width: 
of triangular cross section

height: 
of triangular cross section

asymmetry: 
length of triangular cross section 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::clone "FormFactorLongRipple2Lorentz* FormFactorLongRipple2Lorentz::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::accept "void FormFactorLongRipple2Lorentz::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::getHeight "double FormFactorLongRipple2Lorentz::getHeight() const 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::getWidth "double FormFactorLongRipple2Lorentz::getWidth() const 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::getLength "double FormFactorLongRipple2Lorentz::getLength() const 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::getAsymmetry "double FormFactorLongRipple2Lorentz::getAsymmetry() const 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::getRadialExtension "double FormFactorLongRipple2Lorentz::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorLongRipple2Lorentz::evaluate_for_q "complex_t FormFactorLongRipple2Lorentz::evaluate_for_q(const cvector_t q) const overridefinal

Complex formfactor. 
";


// File: classFormFactorLorentz.xml
%feature("docstring") FormFactorLorentz "

The formfactor of a lorentzian.

C++ includes: FormFactorLorentz.h
";

%feature("docstring")  FormFactorLorentz::FormFactorLorentz "FormFactorLorentz::FormFactorLorentz(double length)
";

%feature("docstring")  FormFactorLorentz::FormFactorLorentz "FormFactorLorentz::FormFactorLorentz(double width, double height)
";

%feature("docstring")  FormFactorLorentz::clone "FormFactorLorentz* FormFactorLorentz::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorLorentz::accept "void FormFactorLorentz::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorLorentz::getWidth "double FormFactorLorentz::getWidth() const 
";

%feature("docstring")  FormFactorLorentz::getHeight "double FormFactorLorentz::getHeight() const 
";

%feature("docstring")  FormFactorLorentz::getRadialExtension "double FormFactorLorentz::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorLorentz::evaluate_for_q "complex_t FormFactorLorentz::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorPolygonalPrism.xml
%feature("docstring") FormFactorPolygonalPrism "

A prism with a polygonal base, for form factor computation.

C++ includes: FormFactorPolyhedron.h
";

%feature("docstring")  FormFactorPolygonalPrism::FormFactorPolygonalPrism "FormFactorPolygonalPrism::FormFactorPolygonalPrism(const double height)
";

%feature("docstring")  FormFactorPolygonalPrism::evaluate_for_q "complex_t FormFactorPolygonalPrism::evaluate_for_q(const cvector_t q) const overridefinal

Returns the form factor F(q) of this polyhedron, respecting the offset height/2. 
";

%feature("docstring")  FormFactorPolygonalPrism::getVolume "double FormFactorPolygonalPrism::getVolume() const overridefinal

Returns the volume of this prism. 
";

%feature("docstring")  FormFactorPolygonalPrism::getHeight "double FormFactorPolygonalPrism::getHeight() const 
";

%feature("docstring")  FormFactorPolygonalPrism::getRadialExtension "double FormFactorPolygonalPrism::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";


// File: classFormFactorPolygonalSurface.xml
%feature("docstring") FormFactorPolygonalSurface "

A polygonal surface, for testing form factor computations.

C++ includes: FormFactorPolyhedron.h
";

%feature("docstring")  FormFactorPolygonalSurface::FormFactorPolygonalSurface "FormFactorPolygonalSurface::FormFactorPolygonalSurface()
";

%feature("docstring")  FormFactorPolygonalSurface::evaluate_for_q "complex_t FormFactorPolygonalSurface::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";

%feature("docstring")  FormFactorPolygonalSurface::getVolume "double FormFactorPolygonalSurface::getVolume() const override

Returns the total volume of the particle of this form factor's shape. 
";

%feature("docstring")  FormFactorPolygonalSurface::getRadialExtension "double FormFactorPolygonalSurface::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";


// File: classFormFactorPolyhedron.xml
%feature("docstring") FormFactorPolyhedron "

A polyhedron, for form factor computation.

C++ includes: FormFactorPolyhedron.h
";

%feature("docstring")  FormFactorPolyhedron::FormFactorPolyhedron "FormFactorPolyhedron::FormFactorPolyhedron()
";

%feature("docstring")  FormFactorPolyhedron::evaluate_for_q "complex_t FormFactorPolyhedron::evaluate_for_q(const cvector_t q) const overridefinal

Returns the form factor F(q) of this polyhedron, respecting the offset z_origin. 
";

%feature("docstring")  FormFactorPolyhedron::evaluate_centered "complex_t FormFactorPolyhedron::evaluate_centered(const cvector_t q) const

Returns the form factor F(q) of this polyhedron, with origin at z=0. 
";

%feature("docstring")  FormFactorPolyhedron::getVolume "double FormFactorPolyhedron::getVolume() const overridefinal

Returns the total volume of the particle of this form factor's shape. 
";

%feature("docstring")  FormFactorPolyhedron::getRadialExtension "double FormFactorPolyhedron::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorPolyhedron::assert_platonic "void FormFactorPolyhedron::assert_platonic() const

Assertions for Platonic solid. 
";


// File: classFormFactorPrism3.xml
%feature("docstring") FormFactorPrism3 "

A prism based on an equilateral triangle.

C++ includes: FormFactorPrism3.h
";

%feature("docstring")  FormFactorPrism3::FormFactorPrism3 "FormFactorPrism3::FormFactorPrism3(const double base_edge, const double height)
";

%feature("docstring")  FormFactorPrism3::clone "FormFactorPrism3* FormFactorPrism3::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorPrism3::accept "void FormFactorPrism3::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorPrism3::getBaseEdge "double FormFactorPrism3::getBaseEdge() const 
";


// File: classFormFactorPrism6.xml
%feature("docstring") FormFactorPrism6 "

A prism based on a regular hexagonal.

C++ includes: FormFactorPrism6.h
";

%feature("docstring")  FormFactorPrism6::FormFactorPrism6 "FormFactorPrism6::FormFactorPrism6(const double base_edge, const double height)
";

%feature("docstring")  FormFactorPrism6::clone "FormFactorPrism6* FormFactorPrism6::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorPrism6::accept "void FormFactorPrism6::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorPrism6::getBaseEdge "double FormFactorPrism6::getBaseEdge() const 
";


// File: classFormFactorPyramid.xml
%feature("docstring") FormFactorPyramid "

A frustum with a quadratic base.

C++ includes: FormFactorPyramid.h
";

%feature("docstring")  FormFactorPyramid::FormFactorPyramid "FormFactorPyramid::FormFactorPyramid(double base_edge, double height, double alpha)

Parameters:
-----------

base_edge: 
of one side of the square base

height: 
of the frustum

alpha: 
dihedral angle in radians between base and facet 
";

%feature("docstring")  FormFactorPyramid::clone "FormFactorPyramid* FormFactorPyramid::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorPyramid::accept "void FormFactorPyramid::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorPyramid::getHeight "double FormFactorPyramid::getHeight() const 
";

%feature("docstring")  FormFactorPyramid::getBaseEdge "double FormFactorPyramid::getBaseEdge() const 
";

%feature("docstring")  FormFactorPyramid::getAlpha "double FormFactorPyramid::getAlpha() const 
";


// File: classFormFactorRipple1.xml
%feature("docstring") FormFactorRipple1 "

The formfactor for a cosine ripple.

C++ includes: FormFactorRipple1.h
";

%feature("docstring")  FormFactorRipple1::FormFactorRipple1 "FormFactorRipple1::FormFactorRipple1(double length, double width, double height)

Ripple1 constructor.

Parameters:
-----------

length: 
of Ripple1

width: 
of cosine cross section

height: 
of cosine cross section 
";

%feature("docstring")  FormFactorRipple1::clone "FormFactorRipple1* FormFactorRipple1::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorRipple1::accept "void FormFactorRipple1::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorRipple1::getLength "double FormFactorRipple1::getLength() const 
";

%feature("docstring")  FormFactorRipple1::getHeight "double FormFactorRipple1::getHeight() const 
";

%feature("docstring")  FormFactorRipple1::getWidth "double FormFactorRipple1::getWidth() const 
";

%feature("docstring")  FormFactorRipple1::getRadialExtension "double FormFactorRipple1::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorRipple1::evaluate_for_q "complex_t FormFactorRipple1::evaluate_for_q(const cvector_t q) const overridefinal

Complex formfactor. 
";


// File: classFormFactorRipple2.xml
%feature("docstring") FormFactorRipple2 "

The formfactor for a triangular ripple.

C++ includes: FormFactorRipple2.h
";

%feature("docstring")  FormFactorRipple2::FormFactorRipple2 "FormFactorRipple2::FormFactorRipple2(double length, double width, double height, double asymmetry)

Ripple2 constructor.

Parameters:
-----------

length: 
of Ripple2

width: 
of triangular cross section

height: 
of triangular cross section

asymmetry: 
length of triangular cross section 
";

%feature("docstring")  FormFactorRipple2::clone "FormFactorRipple2* FormFactorRipple2::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorRipple2::accept "void FormFactorRipple2::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorRipple2::getHeight "double FormFactorRipple2::getHeight() const 
";

%feature("docstring")  FormFactorRipple2::getWidth "double FormFactorRipple2::getWidth() const 
";

%feature("docstring")  FormFactorRipple2::getLength "double FormFactorRipple2::getLength() const 
";

%feature("docstring")  FormFactorRipple2::getAsymmetry "double FormFactorRipple2::getAsymmetry() const 
";

%feature("docstring")  FormFactorRipple2::getRadialExtension "double FormFactorRipple2::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorRipple2::evaluate_for_q "complex_t FormFactorRipple2::evaluate_for_q(const cvector_t q) const overridefinal

Complex formfactor. 
";


// File: classFormFactorSphereGaussianRadius.xml
%feature("docstring") FormFactorSphereGaussianRadius "

A sphere with gaussian radius distribution.

C++ includes: FormFactorSphereGaussianRadius.h
";

%feature("docstring")  FormFactorSphereGaussianRadius::FormFactorSphereGaussianRadius "FormFactorSphereGaussianRadius::FormFactorSphereGaussianRadius(double mean, double sigma)
";

%feature("docstring")  FormFactorSphereGaussianRadius::clone "FormFactorSphereGaussianRadius* FormFactorSphereGaussianRadius::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorSphereGaussianRadius::accept "void FormFactorSphereGaussianRadius::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorSphereGaussianRadius::getRadialExtension "double FormFactorSphereGaussianRadius::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorSphereGaussianRadius::evaluate_for_q "complex_t FormFactorSphereGaussianRadius::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorSphereLogNormalRadius.xml
%feature("docstring") FormFactorSphereLogNormalRadius "

A sphere with log normal radius distribution.

C++ includes: FormFactorSphereLogNormalRadius.h
";

%feature("docstring")  FormFactorSphereLogNormalRadius::FormFactorSphereLogNormalRadius "FormFactorSphereLogNormalRadius::FormFactorSphereLogNormalRadius(double mean, double scale_param, size_t n_samples)
";

%feature("docstring")  FormFactorSphereLogNormalRadius::clone "FormFactorSphereLogNormalRadius* FormFactorSphereLogNormalRadius::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorSphereLogNormalRadius::accept "void FormFactorSphereLogNormalRadius::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorSphereLogNormalRadius::getRadialExtension "double FormFactorSphereLogNormalRadius::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorSphereLogNormalRadius::evaluate_for_q "complex_t FormFactorSphereLogNormalRadius::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorSphereUniformRadius.xml
%feature("docstring") FormFactorSphereUniformRadius "

Integrated full sphere form factor over a uniform distribution of radii.

C++ includes: FormFactorSphereUniformRadius.h
";

%feature("docstring")  FormFactorSphereUniformRadius::FormFactorSphereUniformRadius "FormFactorSphereUniformRadius::FormFactorSphereUniformRadius(double mean, double full_width)
";

%feature("docstring")  FormFactorSphereUniformRadius::clone "FormFactorSphereUniformRadius* FormFactorSphereUniformRadius::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorSphereUniformRadius::accept "void FormFactorSphereUniformRadius::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorSphereUniformRadius::getRadialExtension "double FormFactorSphereUniformRadius::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorSphereUniformRadius::evaluate_for_q "complex_t FormFactorSphereUniformRadius::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorTetrahedron.xml
%feature("docstring") FormFactorTetrahedron "

A frustum with equilateral trigonal base.

C++ includes: FormFactorTetrahedron.h
";

%feature("docstring")  FormFactorTetrahedron::FormFactorTetrahedron "FormFactorTetrahedron::FormFactorTetrahedron(double base_edge, double height, double alpha)

Parameters:
-----------

base_edge: 
of a side of the base

height: 
of the frustum

alpha: 
dihedral angle in radians between base and facet 
";

%feature("docstring")  FormFactorTetrahedron::clone "FormFactorTetrahedron* FormFactorTetrahedron::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorTetrahedron::accept "void FormFactorTetrahedron::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorTetrahedron::getBaseEdge "double FormFactorTetrahedron::getBaseEdge() const 
";

%feature("docstring")  FormFactorTetrahedron::getHeight "double FormFactorTetrahedron::getHeight() const 
";

%feature("docstring")  FormFactorTetrahedron::getAlpha "double FormFactorTetrahedron::getAlpha() const 
";


// File: classFormFactorTriangle.xml
%feature("docstring") FormFactorTriangle "

A planar equilateral triangle, for testing form factor computations.

C++ includes: FormFactorTriangle.h
";

%feature("docstring")  FormFactorTriangle::FormFactorTriangle "FormFactorTriangle::FormFactorTriangle(const double base_edge)
";

%feature("docstring")  FormFactorTriangle::clone "FormFactorTriangle* FormFactorTriangle::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorTriangle::accept "void FormFactorTriangle::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorTriangle::getBaseEdge "double FormFactorTriangle::getBaseEdge() const 
";


// File: classFormFactorTrivial.xml
%feature("docstring") FormFactorTrivial "

A dot, with trivial formfactor F(q)=1.

C++ includes: FormFactorTrivial.h
";

%feature("docstring")  FormFactorTrivial::FormFactorTrivial "FormFactorTrivial::FormFactorTrivial()
";

%feature("docstring")  FormFactorTrivial::clone "FormFactorTrivial* FormFactorTrivial::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorTrivial::accept "void FormFactorTrivial::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorTrivial::getRadialExtension "double FormFactorTrivial::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorTrivial::evaluate_for_q "complex_t FormFactorTrivial::evaluate_for_q(const cvector_t) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorTruncatedCube.xml
%feature("docstring") FormFactorTruncatedCube "

A cube, with tetrahedral truncation of all edges.

C++ includes: FormFactorTruncatedCube.h
";

%feature("docstring")  FormFactorTruncatedCube::FormFactorTruncatedCube "FormFactorTruncatedCube::FormFactorTruncatedCube(double length, double removed_length)

Parameters:
-----------

length: 
of the full cube

removed_length: 
as removed from each edge of the cube 
";

%feature("docstring")  FormFactorTruncatedCube::clone "FormFactorTruncatedCube* FormFactorTruncatedCube::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorTruncatedCube::accept "void FormFactorTruncatedCube::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorTruncatedCube::getLength "double FormFactorTruncatedCube::getLength() const 
";

%feature("docstring")  FormFactorTruncatedCube::getRemovedLength "double FormFactorTruncatedCube::getRemovedLength() const 
";


// File: classFormFactorTruncatedSphere.xml
%feature("docstring") FormFactorTruncatedSphere "

A truncated Sphere.

C++ includes: FormFactorTruncatedSphere.h
";

%feature("docstring")  FormFactorTruncatedSphere::FormFactorTruncatedSphere "FormFactorTruncatedSphere::FormFactorTruncatedSphere(double radius, double height)
";

%feature("docstring")  FormFactorTruncatedSphere::clone "FormFactorTruncatedSphere* FormFactorTruncatedSphere::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorTruncatedSphere::accept "void FormFactorTruncatedSphere::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorTruncatedSphere::getHeight "double FormFactorTruncatedSphere::getHeight() const 
";

%feature("docstring")  FormFactorTruncatedSphere::getRadius "double FormFactorTruncatedSphere::getRadius() const 
";

%feature("docstring")  FormFactorTruncatedSphere::getRadialExtension "double FormFactorTruncatedSphere::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorTruncatedSphere::evaluate_for_q "complex_t FormFactorTruncatedSphere::evaluate_for_q(const cvector_t q) const overridefinal

Complex formfactor. 
";


// File: classFormFactorTruncatedSpheroid.xml
%feature("docstring") FormFactorTruncatedSpheroid "

A truncated spheroid. An ellipsoid with two equal axis, truncated by a plane perpendicular to the third axis.

C++ includes: FormFactorTruncatedSpheroid.h
";

%feature("docstring")  FormFactorTruncatedSpheroid::FormFactorTruncatedSpheroid "FormFactorTruncatedSpheroid::FormFactorTruncatedSpheroid(double radius, double height, double height_flattening)
";

%feature("docstring")  FormFactorTruncatedSpheroid::clone "FormFactorTruncatedSpheroid* FormFactorTruncatedSpheroid::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorTruncatedSpheroid::accept "void FormFactorTruncatedSpheroid::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorTruncatedSpheroid::getRadius "double FormFactorTruncatedSpheroid::getRadius() const 
";

%feature("docstring")  FormFactorTruncatedSpheroid::getHeight "double FormFactorTruncatedSpheroid::getHeight() const 
";

%feature("docstring")  FormFactorTruncatedSpheroid::getHeightFlattening "double FormFactorTruncatedSpheroid::getHeightFlattening() const 
";

%feature("docstring")  FormFactorTruncatedSpheroid::getRadialExtension "double FormFactorTruncatedSpheroid::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorTruncatedSpheroid::evaluate_for_q "complex_t FormFactorTruncatedSpheroid::evaluate_for_q(const cvector_t q) const overridefinal

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classFormFactorWeighted.xml
%feature("docstring") FormFactorWeighted "

Coherent sum of different scalar  IFormFactor's with different weights.

Used by  ParticleComposition and  ParticleCoreShell. If same particles are at different positions, then consider FormFactorDecoratorMultiPositionFactor (restore from commit 0500a26de76).

C++ includes: FormFactorWeighted.h
";

%feature("docstring")  FormFactorWeighted::FormFactorWeighted "FormFactorWeighted::FormFactorWeighted()
";

%feature("docstring")  FormFactorWeighted::~FormFactorWeighted "FormFactorWeighted::~FormFactorWeighted() overridefinal
";

%feature("docstring")  FormFactorWeighted::clone "FormFactorWeighted * FormFactorWeighted::clone() const overridefinal

Returns a clone of this  ISample object. 
";

%feature("docstring")  FormFactorWeighted::accept "void FormFactorWeighted::accept(INodeVisitor *visitor) const overridefinal

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FormFactorWeighted::getRadialExtension "double FormFactorWeighted::getRadialExtension() const overridefinal

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  FormFactorWeighted::addFormFactor "void FormFactorWeighted::addFormFactor(const IFormFactor &form_factor, double weight=1.0)
";

%feature("docstring")  FormFactorWeighted::setAmbientMaterial "void FormFactorWeighted::setAmbientMaterial(const IMaterial &material) overridefinal

Passes the refractive index of the ambient material in which this particle is embedded. 
";

%feature("docstring")  FormFactorWeighted::evaluate "complex_t FormFactorWeighted::evaluate(const WavevectorInfo &wavevectors) const overridefinal

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  FormFactorWeighted::evaluatePol "Eigen::Matrix2cd FormFactorWeighted::evaluatePol(const WavevectorInfo &wavevectors) const overridefinal

Calculates and returns a polarized form factor calculation in DWBA. 
";


// File: classFTDecayFunction1DCauchy.xml
%feature("docstring") FTDecayFunction1DCauchy "

One-dimensional Cauchy decay function in reciprocal space; corresponds to exp(-|x|/omega) in real space.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  FTDecayFunction1DCauchy::FTDecayFunction1DCauchy "FTDecayFunction1DCauchy::FTDecayFunction1DCauchy(double omega)
";

%feature("docstring")  FTDecayFunction1DCauchy::clone "virtual FTDecayFunction1DCauchy* FTDecayFunction1DCauchy::clone() const 
";

%feature("docstring")  FTDecayFunction1DCauchy::accept "void FTDecayFunction1DCauchy::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDecayFunction1DCauchy::evaluate "double FTDecayFunction1DCauchy::evaluate(double q) const final
";


// File: classFTDecayFunction1DGauss.xml
%feature("docstring") FTDecayFunction1DGauss "

One-dimensional Gauss decay function in reciprocal space; corresponds to exp[-x^2/(2*omega^2)] in real space.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  FTDecayFunction1DGauss::FTDecayFunction1DGauss "FTDecayFunction1DGauss::FTDecayFunction1DGauss(double omega)
";

%feature("docstring")  FTDecayFunction1DGauss::clone "virtual FTDecayFunction1DGauss* FTDecayFunction1DGauss::clone() const 
";

%feature("docstring")  FTDecayFunction1DGauss::accept "void FTDecayFunction1DGauss::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDecayFunction1DGauss::evaluate "double FTDecayFunction1DGauss::evaluate(double q) const final
";


// File: classFTDecayFunction1DTriangle.xml
%feature("docstring") FTDecayFunction1DTriangle "

One-dimensional triangle decay function in reciprocal space; corresponds to 1-|x|/omega if |x|<omega (and 0 otherwise) in real space.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  FTDecayFunction1DTriangle::FTDecayFunction1DTriangle "FTDecayFunction1DTriangle::FTDecayFunction1DTriangle(double omega)
";

%feature("docstring")  FTDecayFunction1DTriangle::clone "virtual FTDecayFunction1DTriangle* FTDecayFunction1DTriangle::clone() const 
";

%feature("docstring")  FTDecayFunction1DTriangle::accept "void FTDecayFunction1DTriangle::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDecayFunction1DTriangle::evaluate "double FTDecayFunction1DTriangle::evaluate(double q) const final
";


// File: classFTDecayFunction1DVoigt.xml
%feature("docstring") FTDecayFunction1DVoigt "

One-dimensional pseudo-Voigt decay function in reciprocal space; corresponds to eta*Gauss + (1-eta)*Cauchy.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  FTDecayFunction1DVoigt::FTDecayFunction1DVoigt "FTDecayFunction1DVoigt::FTDecayFunction1DVoigt(double omega, double eta)
";

%feature("docstring")  FTDecayFunction1DVoigt::clone "virtual FTDecayFunction1DVoigt* FTDecayFunction1DVoigt::clone() const 
";

%feature("docstring")  FTDecayFunction1DVoigt::accept "void FTDecayFunction1DVoigt::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDecayFunction1DVoigt::evaluate "double FTDecayFunction1DVoigt::evaluate(double q) const final
";

%feature("docstring")  FTDecayFunction1DVoigt::getEta "double FTDecayFunction1DVoigt::getEta() const 
";


// File: classFTDecayFunction2DCauchy.xml
%feature("docstring") FTDecayFunction2DCauchy "

Two-dimensional Cauchy decay function in reciprocal space; corresponds to exp(-r) in real space, with  $r=\\\\sqrt{(\\\\frac{x}{\\\\omega_x})^2 + (\\\\frac{y}{\\\\omega_y})^2}$.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  FTDecayFunction2DCauchy::FTDecayFunction2DCauchy "FTDecayFunction2DCauchy::FTDecayFunction2DCauchy(double decay_length_x, double decay_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDecayFunction2DCauchy::clone "virtual FTDecayFunction2DCauchy* FTDecayFunction2DCauchy::clone() const 
";

%feature("docstring")  FTDecayFunction2DCauchy::accept "void FTDecayFunction2DCauchy::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDecayFunction2DCauchy::evaluate "double FTDecayFunction2DCauchy::evaluate(double qx, double qy) const final

evaluate Fourier transformed decay function for q in X,Y coordinates 
";


// File: classFTDecayFunction2DGauss.xml
%feature("docstring") FTDecayFunction2DGauss "

Two-dimensional Gauss decay function in reciprocal space; corresponds to exp(-r^2/2) in real space, with  $r=\\\\sqrt{(\\\\frac{x}{\\\\omega_x})^2 + (\\\\frac{y}{\\\\omega_y})^2}$.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  FTDecayFunction2DGauss::FTDecayFunction2DGauss "FTDecayFunction2DGauss::FTDecayFunction2DGauss(double decay_length_x, double decay_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDecayFunction2DGauss::clone "virtual FTDecayFunction2DGauss* FTDecayFunction2DGauss::clone() const 
";

%feature("docstring")  FTDecayFunction2DGauss::accept "void FTDecayFunction2DGauss::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDecayFunction2DGauss::evaluate "double FTDecayFunction2DGauss::evaluate(double qx, double qy) const final

evaluate Fourier transformed decay function for q in X,Y coordinates 
";


// File: classFTDecayFunction2DVoigt.xml
%feature("docstring") FTDecayFunction2DVoigt "

Two-dimensional pseudo-Voigt decay function in reciprocal space; corresponds to eta*Gauss + (1-eta)*Cauchy.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  FTDecayFunction2DVoigt::FTDecayFunction2DVoigt "FTDecayFunction2DVoigt::FTDecayFunction2DVoigt(double decay_length_x, double decay_length_y, double eta, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDecayFunction2DVoigt::clone "virtual FTDecayFunction2DVoigt* FTDecayFunction2DVoigt::clone() const 
";

%feature("docstring")  FTDecayFunction2DVoigt::accept "void FTDecayFunction2DVoigt::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDecayFunction2DVoigt::evaluate "double FTDecayFunction2DVoigt::evaluate(double qx, double qy) const final

evaluate Fourier transformed decay function for q in X,Y coordinates 
";

%feature("docstring")  FTDecayFunction2DVoigt::getEta "virtual double FTDecayFunction2DVoigt::getEta() const 
";


// File: classFTDistribution1DCauchy.xml
%feature("docstring") FTDistribution1DCauchy "

Exponential  IFTDistribution1D exp(-|omega*x|); its Fourier transform evaluate(q) is a Cauchy-Lorentzian starting at evaluate(0)=1.

C++ includes: FTDistributions1D.h
";

%feature("docstring")  FTDistribution1DCauchy::FTDistribution1DCauchy "FTDistribution1DCauchy::FTDistribution1DCauchy(double omega)
";

%feature("docstring")  FTDistribution1DCauchy::clone "FTDistribution1DCauchy* FTDistribution1DCauchy::clone() const final
";

%feature("docstring")  FTDistribution1DCauchy::evaluate "double FTDistribution1DCauchy::evaluate(double q) const final

Returns Fourier transform of this distribution; is a decay function starting at evaluate(0)=1. 
";


// File: classFTDistribution1DCosine.xml
%feature("docstring") FTDistribution1DCosine "

IFTDistribution1D consisting of one cosine wave [1+cos(pi*x/omega) if |x|<omega, and 0 otherwise]; its Fourier transform evaluate(q) starts at evaluate(0)=1.

C++ includes: FTDistributions1D.h
";

%feature("docstring")  FTDistribution1DCosine::FTDistribution1DCosine "FTDistribution1DCosine::FTDistribution1DCosine(double omega)
";

%feature("docstring")  FTDistribution1DCosine::clone "FTDistribution1DCosine* FTDistribution1DCosine::clone() const final
";

%feature("docstring")  FTDistribution1DCosine::evaluate "double FTDistribution1DCosine::evaluate(double q) const final

Returns Fourier transform of this distribution; is a decay function starting at evaluate(0)=1. 
";


// File: classFTDistribution1DGate.xml
%feature("docstring") FTDistribution1DGate "

Square gate  IFTDistribution1D; its Fourier transform evaluate(q) is a sinc function starting at evaluate(0)=1.

C++ includes: FTDistributions1D.h
";

%feature("docstring")  FTDistribution1DGate::FTDistribution1DGate "FTDistribution1DGate::FTDistribution1DGate(double omega)
";

%feature("docstring")  FTDistribution1DGate::clone "FTDistribution1DGate* FTDistribution1DGate::clone() const final
";

%feature("docstring")  FTDistribution1DGate::evaluate "double FTDistribution1DGate::evaluate(double q) const final

Returns Fourier transform of this distribution; is a decay function starting at evaluate(0)=1. 
";


// File: classFTDistribution1DGauss.xml
%feature("docstring") FTDistribution1DGauss "

Gaussian  IFTDistribution1D; its Fourier transform evaluate(q) is a Gaussian starting at evaluate(0)=1.

C++ includes: FTDistributions1D.h
";

%feature("docstring")  FTDistribution1DGauss::FTDistribution1DGauss "FTDistribution1DGauss::FTDistribution1DGauss(double omega)
";

%feature("docstring")  FTDistribution1DGauss::clone "FTDistribution1DGauss* FTDistribution1DGauss::clone() const final
";

%feature("docstring")  FTDistribution1DGauss::evaluate "double FTDistribution1DGauss::evaluate(double q) const final

Returns Fourier transform of this distribution; is a decay function starting at evaluate(0)=1. 
";


// File: classFTDistribution1DTriangle.xml
%feature("docstring") FTDistribution1DTriangle "

Triangle  IFTDistribution1D [1-|x|/omega if |x|<omega, and 0 otherwise]; its Fourier transform evaluate(q) is a squared sinc function starting at evaluate(0)=1.

C++ includes: FTDistributions1D.h
";

%feature("docstring")  FTDistribution1DTriangle::FTDistribution1DTriangle "FTDistribution1DTriangle::FTDistribution1DTriangle(double omega)
";

%feature("docstring")  FTDistribution1DTriangle::~FTDistribution1DTriangle "virtual FTDistribution1DTriangle::~FTDistribution1DTriangle()
";

%feature("docstring")  FTDistribution1DTriangle::clone "FTDistribution1DTriangle* FTDistribution1DTriangle::clone() const final
";

%feature("docstring")  FTDistribution1DTriangle::evaluate "double FTDistribution1DTriangle::evaluate(double q) const final

Returns Fourier transform of this distribution; is a decay function starting at evaluate(0)=1. 
";


// File: classFTDistribution1DVoigt.xml
%feature("docstring") FTDistribution1DVoigt "

IFTDistribution1D that provides a Fourier transform evaluate(q) in form of a pseudo-Voigt decay function eta*Gauss + (1-eta)*Cauchy, with both components starting at 1 for q=0.

C++ includes: FTDistributions1D.h
";

%feature("docstring")  FTDistribution1DVoigt::FTDistribution1DVoigt "FTDistribution1DVoigt::FTDistribution1DVoigt(double omega, double eta)
";

%feature("docstring")  FTDistribution1DVoigt::clone "FTDistribution1DVoigt* FTDistribution1DVoigt::clone() const final
";

%feature("docstring")  FTDistribution1DVoigt::evaluate "double FTDistribution1DVoigt::evaluate(double q) const final

Returns Fourier transform of this distribution; is a decay function starting at evaluate(0)=1. 
";

%feature("docstring")  FTDistribution1DVoigt::getEta "double FTDistribution1DVoigt::getEta() const 
";


// File: classFTDistribution2DCauchy.xml
%feature("docstring") FTDistribution2DCauchy "

Two-dimensional Cauchy distribution in Fourier space; corresponds to a normalized exp(-r) in real space, with  $r=\\\\sqrt{(\\\\frac{x}{\\\\omega_x})^2 + (\\\\frac{y}{\\\\omega_y})^2}$.

C++ includes: FTDistributions2D.h
";

%feature("docstring")  FTDistribution2DCauchy::FTDistribution2DCauchy "FTDistribution2DCauchy::FTDistribution2DCauchy(double coherence_length_x, double coherence_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDistribution2DCauchy::clone "FTDistribution2DCauchy* FTDistribution2DCauchy::clone() const final
";

%feature("docstring")  FTDistribution2DCauchy::accept "void FTDistribution2DCauchy::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDistribution2DCauchy::evaluate "double FTDistribution2DCauchy::evaluate(double qx, double qy) const final

evaluate Fourier transformed distribution for q in X,Y coordinates the original distribution (in real space) is assumed to be normalized: total integral is equal to 1 
";


// File: classFTDistribution2DCone.xml
%feature("docstring") FTDistribution2DCone "

Two-dimensional cone distribution in Fourier space; corresponds to 1-r if r<1 (and 0 otherwise) in real space with  $r=\\\\sqrt{(\\\\frac{x}{\\\\omega_x})^2 + (\\\\frac{y}{\\\\omega_y})^2}$.

C++ includes: FTDistributions2D.h
";

%feature("docstring")  FTDistribution2DCone::FTDistribution2DCone "FTDistribution2DCone::FTDistribution2DCone(double coherence_length_x, double coherence_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDistribution2DCone::clone "FTDistribution2DCone* FTDistribution2DCone::clone() const final
";

%feature("docstring")  FTDistribution2DCone::accept "void FTDistribution2DCone::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDistribution2DCone::evaluate "double FTDistribution2DCone::evaluate(double qx, double qy) const final

evaluate Fourier transformed distribution for q in X,Y coordinates the original distribution (in real space) is assumed to be normalized: total integral is equal to 1 
";


// File: classFTDistribution2DGate.xml
%feature("docstring") FTDistribution2DGate "

Two-dimensional gate distribution in Fourier space; corresponds to normalized constant if r<1 (and 0 otherwise) in real space, with  $r=\\\\sqrt{(\\\\frac{x}{\\\\omega_x})^2 + (\\\\frac{y}{\\\\omega_y})^2}$.

C++ includes: FTDistributions2D.h
";

%feature("docstring")  FTDistribution2DGate::FTDistribution2DGate "FTDistribution2DGate::FTDistribution2DGate(double coherence_length_x, double coherence_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDistribution2DGate::clone "FTDistribution2DGate* FTDistribution2DGate::clone() const final
";

%feature("docstring")  FTDistribution2DGate::accept "void FTDistribution2DGate::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDistribution2DGate::evaluate "double FTDistribution2DGate::evaluate(double qx, double qy) const final

evaluate Fourier transformed distribution for q in X,Y coordinates the original distribution (in real space) is assumed to be normalized: total integral is equal to 1 
";


// File: classFTDistribution2DGauss.xml
%feature("docstring") FTDistribution2DGauss "

Two-dimensional Gauss distribution in Fourier space; corresponds to normalized exp(-r^2/2) in real space with  $r=\\\\sqrt{(\\\\frac{x}{\\\\omega_x})^2 + (\\\\frac{y}{\\\\omega_y})^2}$.

C++ includes: FTDistributions2D.h
";

%feature("docstring")  FTDistribution2DGauss::FTDistribution2DGauss "FTDistribution2DGauss::FTDistribution2DGauss(double coherence_length_x, double coherence_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDistribution2DGauss::clone "FTDistribution2DGauss* FTDistribution2DGauss::clone() const final
";

%feature("docstring")  FTDistribution2DGauss::accept "void FTDistribution2DGauss::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDistribution2DGauss::evaluate "double FTDistribution2DGauss::evaluate(double qx, double qy) const final

evaluate Fourier transformed distribution for q in X,Y coordinates the original distribution (in real space) is assumed to be normalized: total integral is equal to 1 
";


// File: classFTDistribution2DVoigt.xml
%feature("docstring") FTDistribution2DVoigt "

Two-dimensional Voigt distribution in Fourier space; corresponds to eta*Gauss + (1-eta)*Cauchy

C++ includes: FTDistributions2D.h
";

%feature("docstring")  FTDistribution2DVoigt::FTDistribution2DVoigt "FTDistribution2DVoigt::FTDistribution2DVoigt(double coherence_length_x, double coherence_length_y, double eta, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  FTDistribution2DVoigt::clone "FTDistribution2DVoigt* FTDistribution2DVoigt::clone() const final
";

%feature("docstring")  FTDistribution2DVoigt::accept "void FTDistribution2DVoigt::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  FTDistribution2DVoigt::evaluate "double FTDistribution2DVoigt::evaluate(double qx, double qy) const final

evaluate Fourier transformed distribution for q in X,Y coordinates the original distribution (in real space) is assumed to be normalized: total integral is equal to 1 
";

%feature("docstring")  FTDistribution2DVoigt::getEta "double FTDistribution2DVoigt::getEta() const 
";


// File: classFullFresnelMap.xml
%feature("docstring") FullFresnelMap "";

%feature("docstring")  FullFresnelMap::FullFresnelMap "FullFresnelMap::FullFresnelMap()
";

%feature("docstring")  FullFresnelMap::~FullFresnelMap "FullFresnelMap::~FullFresnelMap()
";

%feature("docstring")  FullFresnelMap::push_back "void FullFresnelMap::push_back(ILayerSpecularInfo *layer_map)
";

%feature("docstring")  FullFresnelMap::size "size_t FullFresnelMap::size() const 
";

%feature("docstring")  FullFresnelMap::layerFresnelMap "const ILayerSpecularInfo * FullFresnelMap::layerFresnelMap(size_t index) const 
";


// File: classGISASSimulation.xml
%feature("docstring") GISASSimulation "

Main class to run a Grazing-Incidence Small-Angle Scattering simulation.

C++ includes: GISASSimulation.h
";

%feature("docstring")  GISASSimulation::GISASSimulation "GISASSimulation::GISASSimulation()
";

%feature("docstring")  GISASSimulation::GISASSimulation "GISASSimulation::GISASSimulation(const MultiLayer &p_sample)
";

%feature("docstring")  GISASSimulation::GISASSimulation "GISASSimulation::GISASSimulation(const std::shared_ptr< IMultiLayerBuilder > p_sample_builder)
";

%feature("docstring")  GISASSimulation::~GISASSimulation "GISASSimulation::~GISASSimulation() final
";

%feature("docstring")  GISASSimulation::clone "GISASSimulation* GISASSimulation::clone() const 
";

%feature("docstring")  GISASSimulation::accept "void GISASSimulation::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  GISASSimulation::prepareSimulation "void GISASSimulation::prepareSimulation() final

Put into a clean state for running a simulation. 
";

%feature("docstring")  GISASSimulation::numberOfSimulationElements "int GISASSimulation::numberOfSimulationElements() const final

Gets the number of elements this simulation needs to calculate. 
";

%feature("docstring")  GISASSimulation::getDetectorIntensity "OutputData< double > * GISASSimulation::getDetectorIntensity(IDetector2D::EAxesUnits units_type=IDetector2D::DEFAULT) const

Returns clone of the detector intensity map with detector resolution applied. 
";

%feature("docstring")  GISASSimulation::getIntensityData "Histogram2D * GISASSimulation::getIntensityData(IDetector2D::EAxesUnits units_type=IDetector2D::DEFAULT) const

Returns histogram representing intensity map in requested axes units. 
";

%feature("docstring")  GISASSimulation::setBeamParameters "void GISASSimulation::setBeamParameters(double wavelength, double alpha_i, double phi_i)

Sets beam parameters from here (forwarded to  Instrument) 
";

%feature("docstring")  GISASSimulation::setDetector "void GISASSimulation::setDetector(const IDetector2D &detector)

Sets the detector (axes can be overwritten later) 
";

%feature("docstring")  GISASSimulation::setDetectorParameters "void GISASSimulation::setDetectorParameters(size_t n_phi, double phi_min, double phi_max, size_t n_alpha, double alpha_min, double alpha_max)

Sets spherical detector parameters using angle ranges

Parameters:
-----------

n_phi: 
number of phi-axis bins

phi_min: 
low edge of first phi-bin

phi_max: 
upper edge of last phi-bin

n_alpha: 
number of alpha-axis bins

alpha_min: 
low edge of first alpha-bin

alpha_max: 
upper edge of last alpha-bin 
";

%feature("docstring")  GISASSimulation::removeMasks "void GISASSimulation::removeMasks()

removes all masks from the detector 
";

%feature("docstring")  GISASSimulation::addMask "void GISASSimulation::addMask(const IShape2D &shape, bool mask_value=true)

Adds mask of given shape to the stack of detector masks. The mask value 'true' means that the channel will be excluded from the simulation. The mask which is added last has priority.

Parameters:
-----------

shape: 
The shape of mask ( Rectangle,  Polygon,  Line,  Ellipse)

mask_value: 
The value of mask 
";

%feature("docstring")  GISASSimulation::maskAll "void GISASSimulation::maskAll()

Put the mask for all detector channels (i.e. exclude whole detector from the analysis) 
";

%feature("docstring")  GISASSimulation::setRegionOfInterest "void GISASSimulation::setRegionOfInterest(double xlow, double ylow, double xup, double yup)

Sets rectangular region of interest with lower left and upper right corners defined. 
";

%feature("docstring")  GISASSimulation::resetRegionOfInterest "void GISASSimulation::resetRegionOfInterest()

Resets region of interest making whole detector plane available for the simulation. 
";


// File: classHexagonalLattice.xml
%feature("docstring") HexagonalLattice "";

%feature("docstring")  HexagonalLattice::HexagonalLattice "HexagonalLattice::HexagonalLattice(double length, double rotation_angle=0.0)
";

%feature("docstring")  HexagonalLattice::clone "HexagonalLattice * HexagonalLattice::clone() const 
";

%feature("docstring")  HexagonalLattice::accept "void HexagonalLattice::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  HexagonalLattice::length2 "double HexagonalLattice::length2() const 
";


// File: classHexParaCrystalBuilder.xml
%feature("docstring") HexParaCrystalBuilder "

Builds sample: cylinders with 2DDL structure factor (IsGISAXS example #4).

C++ includes: ParaCrystalBuilder.h
";

%feature("docstring")  HexParaCrystalBuilder::HexParaCrystalBuilder "HexParaCrystalBuilder::HexParaCrystalBuilder()
";

%feature("docstring")  HexParaCrystalBuilder::buildSample "MultiLayer * HexParaCrystalBuilder::buildSample() const 
";


// File: classHistogram1D.xml
%feature("docstring") Histogram1D "

One dimensional histogram.

C++ includes: Histogram1D.h
";

%feature("docstring")  Histogram1D::Histogram1D "Histogram1D::Histogram1D(int nbinsx, double xlow, double xup)

Constructor for fix bin size histograms.

Parameters:
-----------

nbinsx: 
number of bins

xlow: 
low edge of the first bin

xup: 
upper edge of the last bin 
";

%feature("docstring")  Histogram1D::Histogram1D "Histogram1D::Histogram1D(int nbinsx, const std::vector< double > &xbins)

Constructor for variable bin size histograms.

Parameters:
-----------

nbinsx: 
number of bins

xbins: 
Array of size nbins+1 containing low-edges for each bin and upper edge of last bin. 
";

%feature("docstring")  Histogram1D::Histogram1D "Histogram1D::Histogram1D(const IAxis &axis)

Constructor for 1D histogram with custom axis. 
";

%feature("docstring")  Histogram1D::Histogram1D "Histogram1D::Histogram1D(const OutputData< double > &data)

Constructor for 1D histograms from basic  OutputData object. 
";

%feature("docstring")  Histogram1D::clone "Histogram1D * Histogram1D::clone() const

Returns clone of other histogram. 
";

%feature("docstring")  Histogram1D::getRank "size_t Histogram1D::getRank() const

Returns the number of histogram dimensions. 
";

%feature("docstring")  Histogram1D::fill "int Histogram1D::fill(double x, double weight=1.0)

Increment bin with abscissa x with a weight. 
";

%feature("docstring")  Histogram1D::getBinCenters "std::vector< double > Histogram1D::getBinCenters() const

returns vector of histogram bin centers 
";

%feature("docstring")  Histogram1D::getBinValues "std::vector< double > Histogram1D::getBinValues() const

returns vector of bin content (the value accumulated by bins) 
";

%feature("docstring")  Histogram1D::getBinErrors "std::vector< double > Histogram1D::getBinErrors() const

returns vector of bin errors 
";

%feature("docstring")  Histogram1D::crop "Histogram1D * Histogram1D::crop(double xmin, double xmax)

Create new histogram by applying crop on axis. 
";


// File: classHistogram2D.xml
%feature("docstring") Histogram2D "

Two dimensional histogram.

C++ includes: Histogram2D.h
";

%feature("docstring")  Histogram2D::Histogram2D "Histogram2D::Histogram2D(int nbinsx, double xlow, double xup, int nbinsy, double ylow, double yup)

Constructor for fix bin size histograms.

Parameters:
-----------

nbinsx: 
number of bins on X-axis

xlow: 
low edge of the first bin of X-axis

xup: 
upper edge of the last bin of X-axis

nbinsy: 
number of bins on Y axis

ylow: 
low edge of the first bin of Y-axis

yup: 
upper edge of the last bin of Y-axis 
";

%feature("docstring")  Histogram2D::Histogram2D "Histogram2D::Histogram2D(int nbinsx, const std::vector< double > &xbins, int nbinsy, const std::vector< double > &ybins)

Constructor for variable bin size histograms.

Parameters:
-----------

nbinsx: 
number of bins on X-axis

xbins: 
Array of size nbins+1 containing low-edges for each bin and upper edge of last bin.

nbinsy: 
number of bins on Y-axis

ybins: 
Array of size nbins+1 containing low-edges for each bin and upper edge of last bin. 
";

%feature("docstring")  Histogram2D::Histogram2D "Histogram2D::Histogram2D(const IAxis &axis_x, const IAxis &axis_y)

Constructor for 2D histogram with custom axes. 
";

%feature("docstring")  Histogram2D::Histogram2D "Histogram2D::Histogram2D(const OutputData< double > &data)

Constructor for 2D histograms from basic  OutputData object. 
";

%feature("docstring")  Histogram2D::Histogram2D "Histogram2D::Histogram2D(std::vector< std::vector< double >> data)

Constructor for 2D histograms from numpy array (thanks to swig) 
";

%feature("docstring")  Histogram2D::clone "Histogram2D * Histogram2D::clone() const

Returns clone of other histogram. 
";

%feature("docstring")  Histogram2D::getRank "size_t Histogram2D::getRank() const

Returns the number of histogram dimensions. 
";

%feature("docstring")  Histogram2D::fill "int Histogram2D::fill(double x, double y, double weight=1.0)

Increment bin with abscissa x and ordinate y with a weight. 
";

%feature("docstring")  Histogram2D::projectionX "Histogram1D * Histogram2D::projectionX()

Project a 2D histogram into 1D histogram along X. The projection is made from all bins along y-axis. 
";

%feature("docstring")  Histogram2D::projectionX "Histogram1D * Histogram2D::projectionX(double yvalue)

Project a 2D histogram into 1D histogram along X. The projection is made from the y-bin closest to given ordinate yvalue.

Parameters:
-----------

yvalue: 
the value on y-axis at which projection is taken 
";

%feature("docstring")  Histogram2D::projectionX "Histogram1D * Histogram2D::projectionX(double ylow, double yup)

Project a 2D histogram into 1D histogram along X. The projection is made from all y-bins corresponding to ordinate between ylow and yup.

Parameters:
-----------

ylow: 
lower edje on y-axis

yup: 
upper edje on y-axis 
";

%feature("docstring")  Histogram2D::projectionY "Histogram1D * Histogram2D::projectionY()

Project a 2D histogram into 1D histogram along Y. The projection is made from all bins along x-axis. 
";

%feature("docstring")  Histogram2D::projectionY "Histogram1D * Histogram2D::projectionY(double xvalue)

Project a 2D histogram into 1D histogram along Y. The projection is made from the x-bin closest to given abscissa xvalue.

Parameters:
-----------

xvalue: 
the value on x-axis at which projection is taken 
";

%feature("docstring")  Histogram2D::projectionY "Histogram1D * Histogram2D::projectionY(double xlow, double xup)

Project a 2D histogram into 1D histogram along Y. The projection is made from all x-bins corresponding to abscissa between xlow and xup.

Parameters:
-----------

xlow: 
lower edje on x-axis

xup: 
upper edje on x-axis 
";

%feature("docstring")  Histogram2D::crop "Histogram2D * Histogram2D::crop(double xmin, double ymin, double xmax, double ymax)

Create new histogram by applying rectangular clip. 
";

%feature("docstring")  Histogram2D::setContent "void Histogram2D::setContent(const std::vector< std::vector< double >> &data)

Sets the values in histograms channels from numpy array,. 
";

%feature("docstring")  Histogram2D::addContent "void Histogram2D::addContent(const std::vector< std::vector< double >> &data)

Add to values in histograms channels from numpy array,. 
";


// File: classHomogeneousMagneticMaterial.xml
%feature("docstring") HomogeneousMagneticMaterial "

A homogeneous material with magnetization.

C++ includes: HomogeneousMagneticMaterial.h
";

%feature("docstring")  HomogeneousMagneticMaterial::HomogeneousMagneticMaterial "HomogeneousMagneticMaterial::HomogeneousMagneticMaterial(const std::string &name, const complex_t refractive_index, const kvector_t magnetic_field)

Constructs a material with  name,  refractive_index and  magnetic_field
";

%feature("docstring")  HomogeneousMagneticMaterial::HomogeneousMagneticMaterial "HomogeneousMagneticMaterial::HomogeneousMagneticMaterial(const std::string &name, double refractive_index_delta, double refractive_index_beta, const kvector_t magnetic_field)

Constructs a material with  name, refractive_index parameters and  magnetic_field
";

%feature("docstring")  HomogeneousMagneticMaterial::clone "HomogeneousMagneticMaterial * HomogeneousMagneticMaterial::clone() const final
";

%feature("docstring")  HomogeneousMagneticMaterial::cloneInverted "HomogeneousMagneticMaterial * HomogeneousMagneticMaterial::cloneInverted() const final
";

%feature("docstring")  HomogeneousMagneticMaterial::getMagneticField "kvector_t HomogeneousMagneticMaterial::getMagneticField() const

Get the magnetic field (in Tesla) 
";

%feature("docstring")  HomogeneousMagneticMaterial::setMagneticField "void HomogeneousMagneticMaterial::setMagneticField(const kvector_t magnetic_field)

Set the magnetic field (in Tesla) 
";

%feature("docstring")  HomogeneousMagneticMaterial::isScalarMaterial "bool HomogeneousMagneticMaterial::isScalarMaterial() const final

Indicates that the material is not scalar. This means that different polarization states will be diffracted differently 
";

%feature("docstring")  HomogeneousMagneticMaterial::getPolarizedSLD "Eigen::Matrix2cd HomogeneousMagneticMaterial::getPolarizedSLD(const WavevectorInfo &wavevectors) const

Get the scattering matrix (~potential V) from the material. This matrix appears in the full three-dimensional Schroedinger equation. 
";

%feature("docstring")  HomogeneousMagneticMaterial::getPolarizedSLDExperimental "Eigen::Matrix2cd HomogeneousMagneticMaterial::getPolarizedSLDExperimental(const WavevectorInfo &wavevectors) const

Get the scattering matrix for a material defined by its magnetization (experimental) 
";

%feature("docstring")  HomogeneousMagneticMaterial::getPolarizedFresnel "Eigen::Matrix2cd HomogeneousMagneticMaterial::getPolarizedFresnel(const kvector_t k) const

Return the potential term that is used in the one-dimensional Fresnel calculations. 
";

%feature("docstring")  HomogeneousMagneticMaterial::createTransformedMaterial "const IMaterial * HomogeneousMagneticMaterial::createTransformedMaterial(const Transform3D &transform) const final

Create a new material that is transformed with respect to this one. 
";


// File: classHomogeneousMaterial.xml
%feature("docstring") HomogeneousMaterial "

An homogeneous material with a refractive index.

C++ includes: HomogeneousMaterial.h
";

%feature("docstring")  HomogeneousMaterial::HomogeneousMaterial "HomogeneousMaterial::HomogeneousMaterial(const std::string &name, const complex_t refractive_index)

Constructs a material with  name and  refractive_index. 
";

%feature("docstring")  HomogeneousMaterial::HomogeneousMaterial "HomogeneousMaterial::HomogeneousMaterial(const std::string &name, double refractive_index_delta, double refractive_index_beta)

Constructs a material with  name and refractive_index parameters delta and beta (n = 1 - delta + i*beta). 
";

%feature("docstring")  HomogeneousMaterial::~HomogeneousMaterial "virtual HomogeneousMaterial::~HomogeneousMaterial()
";

%feature("docstring")  HomogeneousMaterial::clone "HomogeneousMaterial * HomogeneousMaterial::clone() const 
";

%feature("docstring")  HomogeneousMaterial::cloneInverted "HomogeneousMaterial * HomogeneousMaterial::cloneInverted() const 
";

%feature("docstring")  HomogeneousMaterial::getRefractiveIndex "virtual complex_t HomogeneousMaterial::getRefractiveIndex() const 
";

%feature("docstring")  HomogeneousMaterial::setRefractiveIndex "void HomogeneousMaterial::setRefractiveIndex(const complex_t refractive_index)
";

%feature("docstring")  HomogeneousMaterial::createTransformedMaterial "const IMaterial * HomogeneousMaterial::createTransformedMaterial(const Transform3D &transform) const

Create a new material that is transformed with respect to this one. 
";


// File: classHorizontalLine.xml
%feature("docstring") HorizontalLine "

An infinite horizontal line.

C++ includes: Line.h
";

%feature("docstring")  HorizontalLine::HorizontalLine "HorizontalLine::HorizontalLine(double y)

Parameters:
-----------

y: 
The value at which it crosses y-axes 
";

%feature("docstring")  HorizontalLine::clone "HorizontalLine* HorizontalLine::clone() const 
";

%feature("docstring")  HorizontalLine::contains "bool HorizontalLine::contains(double x, double y) const

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  HorizontalLine::contains "bool HorizontalLine::contains(const Bin1D &binx, const Bin1D &biny) const

Returns true if area defined by two bins is inside or on border of polygon (more precisely, if mid point of two bins satisfy this condition). 
";

%feature("docstring")  HorizontalLine::getYpos "double HorizontalLine::getYpos() const 
";


// File: classIAbstractParticle.xml
%feature("docstring") IAbstractParticle "

Interface for a generic particle.

Inherited by  IParticle and  ParticleDistribution.

C++ includes: IAbstractParticle.h
";

%feature("docstring")  IAbstractParticle::IAbstractParticle "IAbstractParticle::IAbstractParticle()
";

%feature("docstring")  IAbstractParticle::~IAbstractParticle "virtual IAbstractParticle::~IAbstractParticle()
";

%feature("docstring")  IAbstractParticle::clone "virtual IAbstractParticle* IAbstractParticle::clone() const =0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IAbstractParticle::cloneInvertB "virtual IAbstractParticle* IAbstractParticle::cloneInvertB() const =0

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  IAbstractParticle::accept "virtual void IAbstractParticle::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IAbstractParticle::setAmbientMaterial "virtual void IAbstractParticle::setAmbientMaterial(const IMaterial &)=0

Sets the refractive index of the ambient material (which influences its scattering power) 
";

%feature("docstring")  IAbstractParticle::getAbundance "double IAbstractParticle::getAbundance() const 
";

%feature("docstring")  IAbstractParticle::setAbundance "void IAbstractParticle::setAbundance(double abundance)
";

%feature("docstring")  IAbstractParticle::getAmbientMaterial "virtual const IMaterial* IAbstractParticle::getAmbientMaterial() const =0

Returns nullptr, unless overwritten to return a specific material. 
";


// File: classIAxis.xml
%feature("docstring") IAxis "

Interface for one-dimensional axes.

C++ includes: IAxis.h
";

%feature("docstring")  IAxis::IAxis "IAxis::IAxis(const std::string &name)

constructors 
";

%feature("docstring")  IAxis::clone "virtual IAxis* IAxis::clone() const =0

clone function 
";

%feature("docstring")  IAxis::createDoubleBinSize "IAxis * IAxis::createDoubleBinSize() const

Creates a new axis with half the number of bins. 
";

%feature("docstring")  IAxis::~IAxis "virtual IAxis::~IAxis()

destructor 
";

%feature("docstring")  IAxis::size "virtual size_t IAxis::size() const =0

retrieve the number of bins 
";

%feature("docstring")  IAxis::getName "std::string IAxis::getName() const

retrieve the label of the axis 
";

%feature("docstring")  IAxis::setName "void IAxis::setName(std::string name)

Sets the axis label. 
";

%feature("docstring")  IAxis::getBin "virtual Bin1D IAxis::getBin(size_t index) const =0

retrieve a 1d bin for the given index 
";

%feature("docstring")  IAxis::getMin "virtual double IAxis::getMin() const =0

Returns value of first point of axis. 
";

%feature("docstring")  IAxis::getMax "virtual double IAxis::getMax() const =0

Returns value of last point of axis. 
";

%feature("docstring")  IAxis::getBinCenter "virtual double IAxis::getBinCenter(size_t index) const =0
";

%feature("docstring")  IAxis::findClosestIndex "virtual size_t IAxis::findClosestIndex(double value) const =0

find bin index which is best match for given value 
";

%feature("docstring")  IAxis::findIndex "size_t IAxis::findIndex(double value) const

find index of bin that contains the given value returns  size() when value is not found 
";

%feature("docstring")  IAxis::getBinCenters "std::vector< double > IAxis::getBinCenters() const 
";

%feature("docstring")  IAxis::getBinBoundaries "std::vector< double > IAxis::getBinBoundaries() const 
";

%feature("docstring")  IAxis::createClippedAxis "IAxis * IAxis::createClippedAxis(double left, double right) const

Creates a new clipped axis. 
";

%feature("docstring")  IAxis::contains "bool IAxis::contains(double value) const

Returns true if axis contains given point. 
";


// File: classIChiSquaredModule.xml
%feature("docstring") IChiSquaredModule "

Interface for  ChiSquaredModule for chi2 calculations.

Until BornAgain-1.1, there was another child, ChiSquaredFrequency.

C++ includes: IChiSquaredModule.h
";

%feature("docstring")  IChiSquaredModule::IChiSquaredModule "IChiSquaredModule::IChiSquaredModule()
";

%feature("docstring")  IChiSquaredModule::~IChiSquaredModule "IChiSquaredModule::~IChiSquaredModule()
";

%feature("docstring")  IChiSquaredModule::clone "virtual IChiSquaredModule* IChiSquaredModule::clone() const =0

clone method 
";

%feature("docstring")  IChiSquaredModule::getSquaredFunction "const ISquaredFunction* IChiSquaredModule::getSquaredFunction() const

Returns squared function. 
";

%feature("docstring")  IChiSquaredModule::setChiSquaredFunction "void IChiSquaredModule::setChiSquaredFunction(ISquaredFunction *squared_function)

Sets squared function // TODO: merge these two functions (SWIG warning 509) 
";

%feature("docstring")  IChiSquaredModule::setChiSquaredFunction "void IChiSquaredModule::setChiSquaredFunction(const ISquaredFunction &squared_function)
";

%feature("docstring")  IChiSquaredModule::getIntensityNormalizer "virtual const IIntensityNormalizer* IChiSquaredModule::getIntensityNormalizer() const

Returns data normalizer. 
";

%feature("docstring")  IChiSquaredModule::getIntensityNormalizer "virtual IIntensityNormalizer* IChiSquaredModule::getIntensityNormalizer()

Returns data normalizer, non-const version needed to set internals. 
";

%feature("docstring")  IChiSquaredModule::setIntensityNormalizer "void IChiSquaredModule::setIntensityNormalizer(const IIntensityNormalizer &data_normalizer)

Sets data normalizer. 
";

%feature("docstring")  IChiSquaredModule::getIntensityFunction "virtual const IIntensityFunction* IChiSquaredModule::getIntensityFunction() const

Returns data rescaler. 
";

%feature("docstring")  IChiSquaredModule::setIntensityFunction "void IChiSquaredModule::setIntensityFunction(const IIntensityFunction &intensity_function)

Sets data rescaler. 
";

%feature("docstring")  IChiSquaredModule::processFitElements "virtual void IChiSquaredModule::processFitElements(std::vector< FitElement >::iterator, std::vector< FitElement >::iterator)
";


// File: classICloneable.xml
%feature("docstring") ICloneable "

Mix-in for objects that must not be copied, except by cloning.

The base class  INoncopyable disables the copy constructor and the operator= in all its child classes. Child classes of  ICloneable must provide clone().

C++ includes: ICloneable.h
";

%feature("docstring")  ICloneable::clone "virtual ICloneable* ICloneable::clone() const =0
";

%feature("docstring")  ICloneable::transferToCPP "virtual void ICloneable::transferToCPP()

Used for Python overriding of clone (see swig/tweaks.py) 
";


// File: classIClusteredParticles.xml
%feature("docstring") IClusteredParticles "

An ordered assembly of particles. Currently, the only child class is  Crystal.

C++ includes: IClusteredParticles.h
";

%feature("docstring")  IClusteredParticles::IClusteredParticles "IClusteredParticles::IClusteredParticles()
";

%feature("docstring")  IClusteredParticles::clone "virtual IClusteredParticles* IClusteredParticles::clone() const =0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IClusteredParticles::cloneInvertB "virtual IClusteredParticles* IClusteredParticles::cloneInvertB() const =0

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  IClusteredParticles::accept "virtual void IClusteredParticles::accept(INodeVisitor *visitor) const =0

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IClusteredParticles::setAmbientMaterial "virtual void IClusteredParticles::setAmbientMaterial(const IMaterial &material)=0
";

%feature("docstring")  IClusteredParticles::getAmbientMaterial "virtual const IMaterial* IClusteredParticles::getAmbientMaterial() const =0

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  IClusteredParticles::createTotalFormFactor "virtual IFormFactor* IClusteredParticles::createTotalFormFactor(const IFormFactor &, const IRotation *, const kvector_t &) const =0

Creates a total form factor for the mesocrystal with a specific shape and content The bulk content of the mesocrystal is encapsulated by the  IClusteredParticles object itself 
";

%feature("docstring")  IClusteredParticles::applyRotation "virtual void IClusteredParticles::applyRotation(const IRotation &)=delete

Composes transformation with existing one. 
";


// File: classIComputationTerm.xml
%feature("docstring") IComputationTerm "

Computes an independent term of the scattering intensity. Controlled by  MainComputation, which adds up all contributions from subclasses of  IComputationTerm

C++ includes: IComputationTerm.h
";

%feature("docstring")  IComputationTerm::IComputationTerm "IComputationTerm::IComputationTerm(const MultiLayer *p_multilayer)
";

%feature("docstring")  IComputationTerm::~IComputationTerm "IComputationTerm::~IComputationTerm()
";

%feature("docstring")  IComputationTerm::setSpecularInfo "void IComputationTerm::setSpecularInfo(const FullFresnelMap *p_full_map)

Sets magnetic reflection/transmission info for all layers. 
";

%feature("docstring")  IComputationTerm::eval "virtual bool IComputationTerm::eval(const SimulationOptions &options, ProgressHandler *progress, bool polarized, const std::vector< SimulationElement >::iterator &begin_it, const std::vector< SimulationElement >::iterator &end_it) const =0

Calculate scattering intensity for each  SimulationElement returns false if nothing needed to be calculated 
";


// File: classIDetector2D.xml
%feature("docstring") IDetector2D "

Pure virtual detector interface.

C++ includes: IDetector2D.h
";

%feature("docstring")  IDetector2D::IDetector2D "IDetector2D::IDetector2D()
";

%feature("docstring")  IDetector2D::clone "virtual IDetector2D* IDetector2D::clone() const =0
";

%feature("docstring")  IDetector2D::~IDetector2D "IDetector2D::~IDetector2D()
";

%feature("docstring")  IDetector2D::init "virtual void IDetector2D::init(const Beam &)

Inits detector with the beam settings. 
";

%feature("docstring")  IDetector2D::addAxis "void IDetector2D::addAxis(const IAxis &axis)
";

%feature("docstring")  IDetector2D::getAxis "const IAxis & IDetector2D::getAxis(size_t index) const 
";

%feature("docstring")  IDetector2D::getDimension "size_t IDetector2D::getDimension() const 
";

%feature("docstring")  IDetector2D::clear "void IDetector2D::clear()
";

%feature("docstring")  IDetector2D::setDetectorParameters "void IDetector2D::setDetectorParameters(size_t n_x, double x_min, double x_max, size_t n_y, double y_min, double y_max)

Sets detector parameters using angle ranges. 
";

%feature("docstring")  IDetector2D::setDetectorAxes "void IDetector2D::setDetectorAxes(const IAxis &axis0, const IAxis &axis1)

Sets detector parameters using axes. 
";

%feature("docstring")  IDetector2D::setDetectorResolution "void IDetector2D::setDetectorResolution(const IDetectorResolution &p_detector_resolution)

Sets the detector resolution. 
";

%feature("docstring")  IDetector2D::removeDetectorResolution "void IDetector2D::removeDetectorResolution()

Removes detector resolution function. 
";

%feature("docstring")  IDetector2D::applyDetectorResolution "void IDetector2D::applyDetectorResolution(OutputData< double > *p_intensity_map) const

Applies the detector resolution to the given intensity maps. 
";

%feature("docstring")  IDetector2D::getDetectorResolutionFunction "const IDetectorResolution * IDetector2D::getDetectorResolutionFunction() const 
";

%feature("docstring")  IDetector2D::setAnalyzerProperties "void IDetector2D::setAnalyzerProperties(const kvector_t direction, double efficiency, double total_transmission)

Sets the polarization analyzer characteristics of the detector. 
";

%feature("docstring")  IDetector2D::removeMasks "void IDetector2D::removeMasks()

Removes all masks from the detector. 
";

%feature("docstring")  IDetector2D::addMask "void IDetector2D::addMask(const IShape2D &shape, bool mask_value=true)

Adds mask of given shape to the stack of detector masks. The mask value 'true' means that the channel will be excluded from the simulation. The mask which is added last has priority.

Parameters:
-----------

shape: 
The shape of mask ( Rectangle,  Polygon,  Line,  Ellipse)

mask_value: 
The value of mask 
";

%feature("docstring")  IDetector2D::maskAll "void IDetector2D::maskAll()

Put the mask for all detector channels (i.e. exclude whole detector from the analysis) 
";

%feature("docstring")  IDetector2D::getDetectorMask "const DetectorMask * IDetector2D::getDetectorMask() const 
";

%feature("docstring")  IDetector2D::numberOfMaskedChannels "size_t IDetector2D::numberOfMaskedChannels() const 
";

%feature("docstring")  IDetector2D::isMasked "bool IDetector2D::isMasked(size_t index) const 
";

%feature("docstring")  IDetector2D::hasMasks "bool IDetector2D::hasMasks() const

return true if has masks 
";

%feature("docstring")  IDetector2D::createSimulationElements "std::vector< SimulationElement > IDetector2D::createSimulationElements(const Beam &beam)

Create a vector of  SimulationElement objects according to the detector and its mask. 
";

%feature("docstring")  IDetector2D::getSimulationElement "SimulationElement IDetector2D::getSimulationElement(size_t index, const Beam &beam) const

Creates single simulation element. 
";

%feature("docstring")  IDetector2D::createDetectorIntensity "OutputData< double > * IDetector2D::createDetectorIntensity(const std::vector< SimulationElement > &elements, const Beam &beam, IDetector2D::EAxesUnits units_type=IDetector2D::DEFAULT) const

Returns new intensity map with detector resolution applied and axes in requested units. 
";

%feature("docstring")  IDetector2D::createDetectorMap "OutputData< double > * IDetector2D::createDetectorMap(const Beam &beam, EAxesUnits units) const

Returns empty detector map in given axes units. 
";

%feature("docstring")  IDetector2D::initOutputData "void IDetector2D::initOutputData(OutputData< double > &data) const

Inits axes of  OutputData to match the detector and sets values to zero. 
";

%feature("docstring")  IDetector2D::getValidAxesUnits "std::vector< IDetector2D::EAxesUnits > IDetector2D::getValidAxesUnits() const

Returns vector of valid axes units. 
";

%feature("docstring")  IDetector2D::getDefaultAxesUnits "virtual EAxesUnits IDetector2D::getDefaultAxesUnits() const

Return default axes units. 
";

%feature("docstring")  IDetector2D::regionOfInterest "const RegionOfInterest * IDetector2D::regionOfInterest() const

Returns region of interest if exists. 
";

%feature("docstring")  IDetector2D::setRegionOfInterest "void IDetector2D::setRegionOfInterest(double xlow, double ylow, double xup, double yup)

Sets rectangular region of interest with lower left and upper right corners defined. 
";

%feature("docstring")  IDetector2D::resetRegionOfInterest "void IDetector2D::resetRegionOfInterest()

Resets region of interest making whole detector plane available for the simulation. 
";

%feature("docstring")  IDetector2D::getTotalSize "size_t IDetector2D::getTotalSize() const

Returns total number of pixels. 
";

%feature("docstring")  IDetector2D::getAxisBinIndex "size_t IDetector2D::getAxisBinIndex(size_t index, size_t selected_axis) const

Calculate axis index for given global index. 
";

%feature("docstring")  IDetector2D::numberOfSimulationElements "size_t IDetector2D::numberOfSimulationElements() const

Returns number of simulation elements. 
";

%feature("docstring")  IDetector2D::getChildren "std::vector< const INode * > IDetector2D::getChildren() const

Returns a vector of children (const). 
";


// File: classIDetectorResolution.xml
%feature("docstring") IDetectorResolution "

Interface for detector resolution algorithms

C++ includes: IDetectorResolution.h
";

%feature("docstring")  IDetectorResolution::~IDetectorResolution "virtual IDetectorResolution::~IDetectorResolution()
";

%feature("docstring")  IDetectorResolution::applyDetectorResolution "virtual void IDetectorResolution::applyDetectorResolution(OutputData< double > *p_intensity_map) const =0

Apply the resolution function to the intensity data. 
";

%feature("docstring")  IDetectorResolution::applyDetectorResolutionPol "void IDetectorResolution::applyDetectorResolutionPol(OutputData< Eigen::Matrix2d > *p_matrix_intensity) const

Applies the detector resolution to the matrix-valued intensity data. 
";

%feature("docstring")  IDetectorResolution::clone "virtual IDetectorResolution* IDetectorResolution::clone() const =0
";


// File: classIDistribution1D.xml
%feature("docstring") IDistribution1D "

Interface for one-dimensional distributions.

C++ includes: Distributions.h
";

%feature("docstring")  IDistribution1D::IDistribution1D "IDistribution1D::IDistribution1D()
";

%feature("docstring")  IDistribution1D::~IDistribution1D "virtual IDistribution1D::~IDistribution1D()
";

%feature("docstring")  IDistribution1D::clone "virtual IDistribution1D* IDistribution1D::clone() const =0
";

%feature("docstring")  IDistribution1D::probabilityDensity "virtual double IDistribution1D::probabilityDensity(double x) const =0

Returns the distribution-specific probability density for value x. 
";

%feature("docstring")  IDistribution1D::getMean "virtual double IDistribution1D::getMean() const =0

Returns the distribution-specific mean. 
";

%feature("docstring")  IDistribution1D::equidistantSamples "std::vector< ParameterSample > IDistribution1D::equidistantSamples(size_t nbr_samples, double sigma_factor=0., const RealLimits &limits=RealLimits()) const

Returns equidistant samples, using intrinsic parameters, weighted with  probabilityDensity(). 
";

%feature("docstring")  IDistribution1D::equidistantSamplesInRange "std::vector< ParameterSample > IDistribution1D::equidistantSamplesInRange(size_t nbr_samples, double xmin, double xmax) const

Returns equidistant samples from xmin to xmax, weighted with  probabilityDensity(). 
";

%feature("docstring")  IDistribution1D::equidistantPoints "virtual std::vector<double> IDistribution1D::equidistantPoints(size_t nbr_samples, double sigma_factor, const RealLimits &limits=RealLimits()) const =0

Returns equidistant interpolation points, with range computed in distribution-specific way from mean and width parameter, taking into account limits and sigma_factor. 
";

%feature("docstring")  IDistribution1D::equidistantPointsInRange "std::vector< double > IDistribution1D::equidistantPointsInRange(size_t nbr_samples, double xmin, double xmax) const

Returns equidistant interpolation points from xmin to xmax. 
";

%feature("docstring")  IDistribution1D::isDelta "virtual bool IDistribution1D::isDelta() const =0

Returns true if the distribution is in the limit case of a Dirac delta distribution. 
";


// File: classIFactory.xml
%feature("docstring") IFactory "

Base class for all factories.

C++ includes: IFactory.h
";

%feature("docstring")  IFactory::IFactory "IFactory< Key, AbstractProduct >::IFactory()
";

%feature("docstring")  IFactory::createItem "AbstractProduct* IFactory< Key, AbstractProduct >::createItem(const Key &item_key)

Creates object by calling creation function corresponded to given identifier. 
";

%feature("docstring")  IFactory::registerItem "bool IFactory< Key, AbstractProduct >::registerItem(const Key &item_key, CreateItemCallback CreateFn, const std::string &itemDescription=\"\")

Registers object's creation function and store object description. 
";

%feature("docstring")  IFactory::~IFactory "IFactory< Key, AbstractProduct >::~IFactory()
";

%feature("docstring")  IFactory::getNumberOfRegistered "size_t IFactory< Key, AbstractProduct >::getNumberOfRegistered() const

Returns number of registered objects. 
";

%feature("docstring")  IFactory::begin "const_iterator IFactory< Key, AbstractProduct >::begin() const 
";

%feature("docstring")  IFactory::end "const_iterator IFactory< Key, AbstractProduct >::end() const 
";


// File: classIFitObserver.xml
%feature("docstring") IFitObserver "

Base class for all fit observers. Calls update method in following cases: first iteration, every n-th iteration, last iteration and when fit strategy has changed.

C++ includes: IFitObserver.h
";

%feature("docstring")  IFitObserver::IFitObserver "IFitObserver::IFitObserver(int update_every_nth)
";

%feature("docstring")  IFitObserver::notify "void IFitObserver::notify(IObservable *subject)

The method used by Ovservable when he wants to be observed by this. 
";

%feature("docstring")  IFitObserver::update "void IFitObserver::update(FitSuite *fit_suite)

Here Onserver will do actuall job when he thinks that it is a right moment. 
";


// File: classIFitStrategy.xml
%feature("docstring") IFitStrategy "

Interface to concrete fit strategy. Concrete implementation should manipulate with fit parameters/data and then call minimizer.

C++ includes: IFitStrategy.h
";

%feature("docstring")  IFitStrategy::IFitStrategy "IFitStrategy::IFitStrategy(const std::string &name)
";

%feature("docstring")  IFitStrategy::~IFitStrategy "virtual IFitStrategy::~IFitStrategy()
";

%feature("docstring")  IFitStrategy::clone "virtual IFitStrategy* IFitStrategy::clone() const =0
";

%feature("docstring")  IFitStrategy::init "void IFitStrategy::init(FitSuiteImpl *fit_suite)
";

%feature("docstring")  IFitStrategy::execute "virtual void IFitStrategy::execute()=0
";


// File: classIFitSuiteFunction.xml
%feature("docstring") IFitSuiteFunction "

Fitting functions interface to be used by Minimizer.

C++ includes: FitSuiteFunctions.h
";

%feature("docstring")  IFitSuiteFunction::IFitSuiteFunction "IFitSuiteFunction::IFitSuiteFunction()
";

%feature("docstring")  IFitSuiteFunction::~IFitSuiteFunction "virtual IFitSuiteFunction::~IFitSuiteFunction()
";

%feature("docstring")  IFitSuiteFunction::init "virtual void IFitSuiteFunction::init(FitSuiteImpl *fit_suite)
";

%feature("docstring")  IFitSuiteFunction::getNCalls "virtual size_t IFitSuiteFunction::getNCalls() const 
";


// File: classIFormFactor.xml
%feature("docstring") IFormFactor "

Pure virtual base class for all form factors.

The actual form factor is returned by the complex valued function  IFormFactor::evaluate, which depends on the incoming and outgoing wave vectors ki and kf. If it only depends on the scattering vector q=ki-kf, then it is a IBornFormFactor.

Other children besides IBornFormFactor are  IFormFactorDecorator,  FormFactorWeighted,  FormFactorDWBA,  FormFactorDWBAPol and  FormFactorCrystal.

C++ includes: IFormFactor.h
";

%feature("docstring")  IFormFactor::IFormFactor "IFormFactor::IFormFactor()
";

%feature("docstring")  IFormFactor::~IFormFactor "IFormFactor::~IFormFactor() override
";

%feature("docstring")  IFormFactor::clone "IFormFactor* IFormFactor::clone() const override=0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IFormFactor::setAmbientMaterial "virtual void IFormFactor::setAmbientMaterial(const IMaterial &)

Passes the refractive index of the ambient material in which this particle is embedded. 
";

%feature("docstring")  IFormFactor::evaluate "virtual complex_t IFormFactor::evaluate(const WavevectorInfo &wavevectors) const =0

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  IFormFactor::evaluatePol "Eigen::Matrix2cd IFormFactor::evaluatePol(const WavevectorInfo &wavevectors) const

Returns scattering amplitude for matrix interactions. 
";

%feature("docstring")  IFormFactor::getVolume "double IFormFactor::getVolume() const

Returns the total volume of the particle of this form factor's shape. 
";

%feature("docstring")  IFormFactor::getRadialExtension "virtual double IFormFactor::getRadialExtension() const =0

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";

%feature("docstring")  IFormFactor::setSpecularInfo "virtual void IFormFactor::setSpecularInfo(const ILayerRTCoefficients *, const ILayerRTCoefficients *)

Sets reflection/transmission info. 
";


// File: classIFormFactorBorn.xml
%feature("docstring") IFormFactorBorn "

Pure virtual base class for Born form factors.

In contrast to the generic  IFormFactor, a Born form factor does not depend on the incoming and outgoing wave vectors ki and kf, except through their difference, the scattering vector q=ki-kf.

C++ includes: IFormFactorBorn.h
";

%feature("docstring")  IFormFactorBorn::IFormFactorBorn "IFormFactorBorn::IFormFactorBorn()
";

%feature("docstring")  IFormFactorBorn::~IFormFactorBorn "IFormFactorBorn::~IFormFactorBorn() override
";

%feature("docstring")  IFormFactorBorn::clone "IFormFactorBorn* IFormFactorBorn::clone() const override=0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IFormFactorBorn::evaluate "complex_t IFormFactorBorn::evaluate(const WavevectorInfo &wavevectors) const override

Returns scattering amplitude for complex wavevectors ki, kf. 
";

%feature("docstring")  IFormFactorBorn::evaluatePol "Eigen::Matrix2cd IFormFactorBorn::evaluatePol(const WavevectorInfo &wavevectors) const override

Returns scattering amplitude for matrix interactions. 
";

%feature("docstring")  IFormFactorBorn::evaluate_for_q "virtual complex_t IFormFactorBorn::evaluate_for_q(const cvector_t q) const =0

Returns scattering amplitude for complex scattering wavevector q=k_i-k_f. This method is public only for convenience of plotting form factors in Python. 
";


// File: classIFormFactorDecorator.xml
%feature("docstring") IFormFactorDecorator "

Encapsulates another formfactor and adds extra functionality (a scalar factor, a Debye-Waller factor, a position-dependent phase factor, ...).

This class is designed according to the Decorator Pattern. It inherits from  IFormFactor and has a member of type IFormFactor*.

C++ includes: IFormFactorDecorator.h
";

%feature("docstring")  IFormFactorDecorator::IFormFactorDecorator "IFormFactorDecorator::IFormFactorDecorator(const IFormFactor &form_factor)
";

%feature("docstring")  IFormFactorDecorator::~IFormFactorDecorator "IFormFactorDecorator::~IFormFactorDecorator() override
";

%feature("docstring")  IFormFactorDecorator::clone "IFormFactorDecorator* IFormFactorDecorator::clone() const override=0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IFormFactorDecorator::accept "void IFormFactorDecorator::accept(INodeVisitor *visitor) const override=0

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IFormFactorDecorator::setAmbientMaterial "void IFormFactorDecorator::setAmbientMaterial(const IMaterial &material) override

Passes the refractive index of the ambient material in which this particle is embedded. 
";

%feature("docstring")  IFormFactorDecorator::getVolume "double IFormFactorDecorator::getVolume() const override

Returns the total volume of the particle of this form factor's shape. 
";

%feature("docstring")  IFormFactorDecorator::getRadialExtension "double IFormFactorDecorator::getRadialExtension() const override

Returns the (approximate in some cases) radial size of the particle of this form factor's shape. This is used for SSCA calculations 
";


// File: classIFTDecayFunction1D.xml
%feature("docstring") IFTDecayFunction1D "

Interface for a one-dimensional decay function, with evaluate(q) returning the Fourier transform, normalized to  $\\\\int dq\\\\; {\\\\rm evaluate}(q) = 1$.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  IFTDecayFunction1D::IFTDecayFunction1D "IFTDecayFunction1D::IFTDecayFunction1D(double omega)
";

%feature("docstring")  IFTDecayFunction1D::clone "virtual IFTDecayFunction1D* IFTDecayFunction1D::clone() const =0
";

%feature("docstring")  IFTDecayFunction1D::evaluate "virtual double IFTDecayFunction1D::evaluate(double q) const =0
";

%feature("docstring")  IFTDecayFunction1D::setOmega "void IFTDecayFunction1D::setOmega(double omega)
";

%feature("docstring")  IFTDecayFunction1D::getOmega "double IFTDecayFunction1D::getOmega() const 
";


// File: classIFTDecayFunction2D.xml
%feature("docstring") IFTDecayFunction2D "

Interface for two-dimensional decay function in reciprocal space.

C++ includes: FTDecayFunctions.h
";

%feature("docstring")  IFTDecayFunction2D::IFTDecayFunction2D "IFTDecayFunction2D::IFTDecayFunction2D(double decay_length_x, double decay_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  IFTDecayFunction2D::clone "virtual IFTDecayFunction2D* IFTDecayFunction2D::clone() const =0
";

%feature("docstring")  IFTDecayFunction2D::setGamma "void IFTDecayFunction2D::setGamma(double gamma)

set angle between first lattice vector and X-axis of distribution (both in direct space) 
";

%feature("docstring")  IFTDecayFunction2D::getGamma "double IFTDecayFunction2D::getGamma() const

get angle between first lattice vector and X-axis of distribution (both in direct space) 
";

%feature("docstring")  IFTDecayFunction2D::getDelta "double IFTDecayFunction2D::getDelta() const

get angle between X- and Y-axis of distribution (in direct space) 
";

%feature("docstring")  IFTDecayFunction2D::getDecayLengthX "double IFTDecayFunction2D::getDecayLengthX() const

get coherence length in X-direction 
";

%feature("docstring")  IFTDecayFunction2D::getDecayLengthY "double IFTDecayFunction2D::getDecayLengthY() const

get coherence length in Y-direction 
";

%feature("docstring")  IFTDecayFunction2D::evaluate "virtual double IFTDecayFunction2D::evaluate(double qx, double qy) const =0

evaluate Fourier transformed decay function for q in X,Y coordinates 
";

%feature("docstring")  IFTDecayFunction2D::transformToStarBasis "void IFTDecayFunction2D::transformToStarBasis(double qX, double qY, double alpha, double a, double b, double &qa, double &qb) const

transform back to a*, b* basis: 
";


// File: classIFTDistribution1D.xml
%feature("docstring") IFTDistribution1D "

Interface for a one-dimensional distribution, with normalization adjusted so that the Fourier transform evaluate(q) is a decay function that starts at evaluate(0)=1.

C++ includes: FTDistributions1D.h
";

%feature("docstring")  IFTDistribution1D::IFTDistribution1D "IFTDistribution1D::IFTDistribution1D(double omega)
";

%feature("docstring")  IFTDistribution1D::clone "virtual IFTDistribution1D* IFTDistribution1D::clone() const =0
";

%feature("docstring")  IFTDistribution1D::evaluate "virtual double IFTDistribution1D::evaluate(double q) const =0

Returns Fourier transform of this distribution; is a decay function starting at evaluate(0)=1. 
";

%feature("docstring")  IFTDistribution1D::setOmega "void IFTDistribution1D::setOmega(double omega)
";

%feature("docstring")  IFTDistribution1D::getOmega "double IFTDistribution1D::getOmega() const 
";


// File: classIFTDistribution2D.xml
%feature("docstring") IFTDistribution2D "

Interface for two-dimensional distributions in Fourier space.

C++ includes: FTDistributions2D.h
";

%feature("docstring")  IFTDistribution2D::IFTDistribution2D "IFTDistribution2D::IFTDistribution2D(double coherence_length_x, double coherence_length_y, double gamma=0, double delta=M_PI_2)
";

%feature("docstring")  IFTDistribution2D::clone "IFTDistribution2D* IFTDistribution2D::clone() const =0
";

%feature("docstring")  IFTDistribution2D::setGamma "void IFTDistribution2D::setGamma(double gamma)
";

%feature("docstring")  IFTDistribution2D::getGamma "double IFTDistribution2D::getGamma() const 
";

%feature("docstring")  IFTDistribution2D::getDelta "double IFTDistribution2D::getDelta() const 
";

%feature("docstring")  IFTDistribution2D::getCoherenceLengthX "double IFTDistribution2D::getCoherenceLengthX() const 
";

%feature("docstring")  IFTDistribution2D::getCoherenceLengthY "double IFTDistribution2D::getCoherenceLengthY() const 
";

%feature("docstring")  IFTDistribution2D::evaluate "virtual double IFTDistribution2D::evaluate(double qx, double qy) const =0

evaluate Fourier transformed distribution for q in X,Y coordinates the original distribution (in real space) is assumed to be normalized: total integral is equal to 1 
";


// File: classIHistogram.xml
%feature("docstring") IHistogram "

Base class for 1D and 2D histograms holding values of double type.

C++ includes: IHistogram.h
";

%feature("docstring")  IHistogram::IHistogram "IHistogram::IHistogram()
";

%feature("docstring")  IHistogram::IHistogram "IHistogram::IHistogram(const IHistogram &other)
";

%feature("docstring")  IHistogram::~IHistogram "virtual IHistogram::~IHistogram()
";

%feature("docstring")  IHistogram::IHistogram "IHistogram::IHistogram(const IAxis &axis_x)
";

%feature("docstring")  IHistogram::IHistogram "IHistogram::IHistogram(const IAxis &axis_x, const IAxis &axis_y)
";

%feature("docstring")  IHistogram::clone "virtual IHistogram* IHistogram::clone() const =0
";

%feature("docstring")  IHistogram::getRank "virtual size_t IHistogram::getRank() const =0

Returns number of histogram dimensions. 
";

%feature("docstring")  IHistogram::getTotalNumberOfBins "size_t IHistogram::getTotalNumberOfBins() const

Returns total number of histogram bins. For 2D histograms the result will be the product of bin numbers along X and Y axes. 
";

%feature("docstring")  IHistogram::getXaxis "const IAxis & IHistogram::getXaxis() const

returns x-axis 
";

%feature("docstring")  IHistogram::getYaxis "const IAxis & IHistogram::getYaxis() const

returns y-axis for 2D histograms 
";

%feature("docstring")  IHistogram::getXmin "double IHistogram::getXmin() const

Returns x-axis min (lower edge of first bin). 
";

%feature("docstring")  IHistogram::getXmax "double IHistogram::getXmax() const

Returns x-axis max (upper edge of last bin). 
";

%feature("docstring")  IHistogram::getNbinsX "size_t IHistogram::getNbinsX() const

Returns number of bins on x-axis. 
";

%feature("docstring")  IHistogram::getYmin "double IHistogram::getYmin() const

Returns y-axis min (lower edge of first bin) for 2D histograms. 
";

%feature("docstring")  IHistogram::getYmax "double IHistogram::getYmax() const

Returns y-axis max (upper edge of last bin) for 2D histograms. 
";

%feature("docstring")  IHistogram::getNbinsY "size_t IHistogram::getNbinsY() const

Returns number of bins on y-axis. 
";

%feature("docstring")  IHistogram::getGlobalBin "size_t IHistogram::getGlobalBin(size_t binx, size_t biny=0) const

Returns global bin index for given axes indices. For 1D histogram the global bin index coinside with x-axis index.

Parameters:
-----------

binx: 
x-axis bin index

biny: 
y-axis bin index (for 2D histograms)

The global bin index 
";

%feature("docstring")  IHistogram::findGlobalBin "size_t IHistogram::findGlobalBin(double x, double y) const

Returns closest global bin index for given axes coordinates. For 1D.

Parameters:
-----------

x: 
Value on x-axis

y: 
Value on y-axis (for 2D histograms)

Closest global bin index 
";

%feature("docstring")  IHistogram::getXaxisIndex "int IHistogram::getXaxisIndex(size_t i) const

Returns x-axis bin index for given globalbin. For 1D histograms returned value conicide with globalbin value. 
";

%feature("docstring")  IHistogram::getYaxisIndex "int IHistogram::getYaxisIndex(size_t i) const

Returns y-axis bin index for given globalbin (for 2D histograms). 
";

%feature("docstring")  IHistogram::getXaxisValue "double IHistogram::getXaxisValue(size_t i)

Returns the value on x-axis corresponding to the global bin index.

Parameters:
-----------

globalbin: 
The global bin index

The center of corresponding bin of the axis 
";

%feature("docstring")  IHistogram::getYaxisValue "double IHistogram::getYaxisValue(size_t i)

Returns the value on y-axis corresponding to the 2D histogram global bin index.

Parameters:
-----------

globalbin: 
The global bin index

The center of corresponding bin of the axis 
";

%feature("docstring")  IHistogram::getBinContent "double IHistogram::getBinContent(size_t i) const

Returns content (accumulated value) of the bin with given index.

Parameters:
-----------

globalbin: 
The global bin index

The value accumulated by the bin (integral) 
";

%feature("docstring")  IHistogram::getData "const OutputData<CumulativeValue>& IHistogram::getData() const 
";

%feature("docstring")  IHistogram::getData "OutputData<CumulativeValue>& IHistogram::getData()
";

%feature("docstring")  IHistogram::getBinContent "double IHistogram::getBinContent(size_t binx, size_t biny) const

Returns content (accumulated value) of the 2D histogram bin with given indices.

Parameters:
-----------

binx: 
x-axis bin index

biny: 
y-axis bin index

The value accumulated by the bin (integral) 
";

%feature("docstring")  IHistogram::setBinContent "void IHistogram::setBinContent(size_t i, double value)

Sets content of the bin corresponding to the globalbin number. 
";

%feature("docstring")  IHistogram::addBinContent "void IHistogram::addBinContent(size_t i, double value)

Add the value to the bin. 
";

%feature("docstring")  IHistogram::getBinError "double IHistogram::getBinError(size_t i) const

Returns error of the bin with given index. 
";

%feature("docstring")  IHistogram::getBinError "double IHistogram::getBinError(size_t binx, size_t biny) const

Returns error of the bin with given indices (for 2D histograms). 
";

%feature("docstring")  IHistogram::getBinAverage "double IHistogram::getBinAverage(size_t i) const

Returns average value in the bin with given index. 
";

%feature("docstring")  IHistogram::getBinAverage "double IHistogram::getBinAverage(size_t binx, size_t biny) const

Returns average value of the bin with given indices (for 2D histograms). 
";

%feature("docstring")  IHistogram::getBinNumberOfEntries "int IHistogram::getBinNumberOfEntries(size_t i) const

Returns number of entries in the bin with given index. 
";

%feature("docstring")  IHistogram::getBinNumberOfEntries "int IHistogram::getBinNumberOfEntries(size_t binx, size_t biny) const

Returns number of entries in the bin with given indices (for 2D histograms). 
";

%feature("docstring")  IHistogram::getMaximum "double IHistogram::getMaximum() const

Returns histogram maximum value (maximum of  getBinContent() over all bins) 
";

%feature("docstring")  IHistogram::getMaximumBinIndex "size_t IHistogram::getMaximumBinIndex() const

Returns globalbin index with maximum content. 
";

%feature("docstring")  IHistogram::getMinimum "double IHistogram::getMinimum() const

Returns histogram minimum value (minimum of  getBinContent() over all bins) 
";

%feature("docstring")  IHistogram::getMinimumBinIndex "size_t IHistogram::getMinimumBinIndex() const

Returns globalbin index with minimum content. 
";

%feature("docstring")  IHistogram::scale "void IHistogram::scale(double value)

Multiply this histogram (every bin content value) by a constant. 
";

%feature("docstring")  IHistogram::integral "double IHistogram::integral() const

Returns integral of bins content (computed as a sum of all bin content). 
";

%feature("docstring")  IHistogram::reset "void IHistogram::reset()

Reset histogram content (axes remains) 
";

%feature("docstring")  IHistogram::createOutputData "OutputData< double > * IHistogram::createOutputData(DataType dataType=DataType::INTEGRAL) const

creates new  OutputData with histogram's shape and values corresponding to DataType

creates new  OutputData with histogram's shape and put there values corresponding to DataType 
";

%feature("docstring")  IHistogram::hasSameShape "bool IHistogram::hasSameShape(const IHistogram &other) const

Returns true if objects a) have same dimensions b) bin boundaries of axes coincide. 
";

%feature("docstring")  IHistogram::hasSameDimensions "bool IHistogram::hasSameDimensions(const IHistogram &other) const

Returns true if object have same rank and number of axes bins. 
";

%feature("docstring")  IHistogram::relativeDifferenceHistogram "IHistogram * IHistogram::relativeDifferenceHistogram(const IHistogram &rhs)

returns histogram representing relative difference of two histograms. 
";

%feature("docstring")  IHistogram::save "void IHistogram::save(const std::string &filename)

Saves histogram in file Following formats are available: *.txt, *.tif, *.int (*.txt.gz, *.tif.gz, *.int.gz) 
";

%feature("docstring")  IHistogram::load "void IHistogram::load(const std::string &filename)

Loads histogram from file, the shape of array in file should match Following formats are available: *.txt, *.tif, *.int (*.txt.gz, *.tif.gz, *.int.gz) Only bin content will be loaded, histogram axes remain the same. 
";


// File: classIIntensityFunction.xml
%feature("docstring") IIntensityFunction "

Interface for applying arbitrary function to the measured intensity.

C++ includes: IIntensityFunction.h
";

%feature("docstring")  IIntensityFunction::~IIntensityFunction "virtual IIntensityFunction::~IIntensityFunction()
";

%feature("docstring")  IIntensityFunction::clone "virtual IIntensityFunction* IIntensityFunction::clone() const =0
";

%feature("docstring")  IIntensityFunction::evaluate "virtual double IIntensityFunction::evaluate(double value) const =0
";


// File: classIIntensityNormalizer.xml
%feature("docstring") IIntensityNormalizer "

Interface to  OutputData normalizers.

C++ includes: IIntensityNormalizer.h
";

%feature("docstring")  IIntensityNormalizer::~IIntensityNormalizer "virtual IIntensityNormalizer::~IIntensityNormalizer()
";

%feature("docstring")  IIntensityNormalizer::clone "virtual IIntensityNormalizer* IIntensityNormalizer::clone() const =0
";

%feature("docstring")  IIntensityNormalizer::createNormalizedData "virtual OutputData<double>* IIntensityNormalizer::createNormalizedData(const OutputData< double > &data) const =0
";

%feature("docstring")  IIntensityNormalizer::apply "virtual void IIntensityNormalizer::apply(OutputData< double > &data) const =0
";

%feature("docstring")  IIntensityNormalizer::setMaximumIntensity "virtual void IIntensityNormalizer::setMaximumIntensity(double)=0
";


// File: classIInterferenceFunction.xml
%feature("docstring") IInterferenceFunction "

Pure virtual base class of interference functions.

C++ includes: IInterferenceFunction.h
";

%feature("docstring")  IInterferenceFunction::~IInterferenceFunction "virtual IInterferenceFunction::~IInterferenceFunction()
";

%feature("docstring")  IInterferenceFunction::clone "virtual IInterferenceFunction* IInterferenceFunction::clone() const =0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IInterferenceFunction::accept "virtual void IInterferenceFunction::accept(INodeVisitor *visitor) const =0

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IInterferenceFunction::evaluate "virtual double IInterferenceFunction::evaluate(const kvector_t q) const =0

Evaluates the interference function for a given wavevector transfer (only the real x and y components are relevant) 
";

%feature("docstring")  IInterferenceFunction::getKappa "virtual double IInterferenceFunction::getKappa() const

Retrieves the size-distance coupling constant (default 0.0) 
";

%feature("docstring")  IInterferenceFunction::getParticleDensity "virtual double IInterferenceFunction::getParticleDensity() const

If defined by this interference function's parameters, returns the particle density (per area). Otherwise, returns zero or a user-defined value 
";


// File: classIInterferenceFunctionStrategy.xml
%feature("docstring") IInterferenceFunctionStrategy "

Pure virtual base class of all interference function strategy classes. Provides an 'evaluate' function that computes the total scattering intensity from a decorated layer, taking into account a specific inter-particle interference function. This function uses low-level functions precomputeParticleFormfactors, evaluateForList that are implemented differently in different inheriting classes. Multiple inheritance is used to support scalar and polarized scattering (through  IInterferenceFunctionStrategy1,  IInterferenceFunctionStrategy2) and to implement different approximation schemes ( DecouplingApproximationStrategy1,  SSCApproximationStrategy1, and their polarized counterparts).

Instantiation of child classes takes place in  LayerStrategyBuilder::createStrategy, which is called from  ParticleLayoutComputation::eval.

C++ includes: IInterferenceFunctionStrategy.h
";

%feature("docstring")  IInterferenceFunctionStrategy::IInterferenceFunctionStrategy "IInterferenceFunctionStrategy::IInterferenceFunctionStrategy()
";

%feature("docstring")  IInterferenceFunctionStrategy::IInterferenceFunctionStrategy "IInterferenceFunctionStrategy::IInterferenceFunctionStrategy(const SimulationOptions &sim_params)
";

%feature("docstring")  IInterferenceFunctionStrategy::~IInterferenceFunctionStrategy "IInterferenceFunctionStrategy::~IInterferenceFunctionStrategy()
";

%feature("docstring")  IInterferenceFunctionStrategy::init "void IInterferenceFunctionStrategy::init(const SafePointerVector< FormFactorCoherentSum > &weighted_formfactors, const IInterferenceFunction *p_iff)

Initializes the object with form factors and interference functions. 
";

%feature("docstring")  IInterferenceFunctionStrategy::evaluate "double IInterferenceFunctionStrategy::evaluate(const SimulationElement &sim_element) const

Calculates the intensity for scalar particles/interactions. 
";


// File: classIInterferenceFunctionStrategy1.xml
%feature("docstring") IInterferenceFunctionStrategy1 "

Pure virtual base class of all scalar interference function strategy classes. Provides the precomputation of particle form factors.

C++ includes: IInterferenceFunctionStrategy.h
";


// File: classIInterferenceFunctionStrategy2.xml
%feature("docstring") IInterferenceFunctionStrategy2 "

Pure virtual base class of all polarized interference function strategy classes. Provides the precomputation of particle form factors.

C++ includes: IInterferenceFunctionStrategy.h
";


// File: classILayerRTCoefficients.xml
%feature("docstring") ILayerRTCoefficients "

Interface to access reflection/transmission coefficients.

C++ includes: ILayerRTCoefficients.h
";

%feature("docstring")  ILayerRTCoefficients::~ILayerRTCoefficients "virtual ILayerRTCoefficients::~ILayerRTCoefficients()
";

%feature("docstring")  ILayerRTCoefficients::clone "virtual ILayerRTCoefficients* ILayerRTCoefficients::clone() const =0
";

%feature("docstring")  ILayerRTCoefficients::T1plus "virtual Eigen::Vector2cd ILayerRTCoefficients::T1plus() const =0

The following functions return the transmitted and reflected amplitudes for different incoming beam polarizations and eigenmodes 
";

%feature("docstring")  ILayerRTCoefficients::R1plus "virtual Eigen::Vector2cd ILayerRTCoefficients::R1plus() const =0
";

%feature("docstring")  ILayerRTCoefficients::T2plus "virtual Eigen::Vector2cd ILayerRTCoefficients::T2plus() const =0
";

%feature("docstring")  ILayerRTCoefficients::R2plus "virtual Eigen::Vector2cd ILayerRTCoefficients::R2plus() const =0
";

%feature("docstring")  ILayerRTCoefficients::T1min "virtual Eigen::Vector2cd ILayerRTCoefficients::T1min() const =0
";

%feature("docstring")  ILayerRTCoefficients::R1min "virtual Eigen::Vector2cd ILayerRTCoefficients::R1min() const =0
";

%feature("docstring")  ILayerRTCoefficients::T2min "virtual Eigen::Vector2cd ILayerRTCoefficients::T2min() const =0
";

%feature("docstring")  ILayerRTCoefficients::R2min "virtual Eigen::Vector2cd ILayerRTCoefficients::R2min() const =0
";

%feature("docstring")  ILayerRTCoefficients::getKz "virtual Eigen::Vector2cd ILayerRTCoefficients::getKz() const =0

Returns z-part of the two wavevector eigenmodes. 
";

%feature("docstring")  ILayerRTCoefficients::getScalarT "virtual complex_t ILayerRTCoefficients::getScalarT() const

Scalar value getters; these throw errors by default as they should only be used when the derived object is really scalar 
";

%feature("docstring")  ILayerRTCoefficients::getScalarR "virtual complex_t ILayerRTCoefficients::getScalarR() const 
";

%feature("docstring")  ILayerRTCoefficients::getScalarKz "virtual complex_t ILayerRTCoefficients::getScalarKz() const 
";


// File: classILayerSpecularInfo.xml
%feature("docstring") ILayerSpecularInfo "

Holds the necessary information to calculate the radiation wavefunction in a specific layer for different incoming (outgoing) angles of the beam in the top layer (these amplitudes correspond to the specular part of the wavefunction).

C++ includes: ILayerSpecularInfo.h
";

%feature("docstring")  ILayerSpecularInfo::ILayerSpecularInfo "ILayerSpecularInfo::ILayerSpecularInfo()
";

%feature("docstring")  ILayerSpecularInfo::~ILayerSpecularInfo "ILayerSpecularInfo::~ILayerSpecularInfo()
";

%feature("docstring")  ILayerSpecularInfo::clone "ILayerSpecularInfo* ILayerSpecularInfo::clone() const =0
";

%feature("docstring")  ILayerSpecularInfo::getOutCoefficients "virtual const ILayerRTCoefficients* ILayerSpecularInfo::getOutCoefficients(const SimulationElement &sim_element) const =0

Retrieves the amplitude coefficients for a (time-reversed) outgoing wavevector. 
";

%feature("docstring")  ILayerSpecularInfo::getInCoefficients "virtual const ILayerRTCoefficients* ILayerSpecularInfo::getInCoefficients(const SimulationElement &sim_element) const =0

Retrieves the amplitude coefficients for an incoming wavevector. 
";


// File: classILayout.xml
%feature("docstring") ILayout "

Pure virtual interface class to equip a sample layer with scattering properties. Currently only inherited by  ParticleLayout; in the future also by domain structure.

C++ includes: ILayout.h
";

%feature("docstring")  ILayout::ILayout "ILayout::ILayout()
";

%feature("docstring")  ILayout::~ILayout "virtual ILayout::~ILayout()
";

%feature("docstring")  ILayout::clone "virtual ILayout* ILayout::clone() const =0

Returns a clone of this  ISample object. 
";

%feature("docstring")  ILayout::accept "virtual void ILayout::accept(INodeVisitor *visitor) const =0

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  ILayout::cloneInvertB "virtual ILayout* ILayout::cloneInvertB() const =0

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  ILayout::getNumberOfParticles "virtual size_t ILayout::getNumberOfParticles() const =0

Returns number of particles. 
";

%feature("docstring")  ILayout::getParticle "virtual const IAbstractParticle* ILayout::getParticle(size_t index) const =0

Returns information about particle with index. 
";

%feature("docstring")  ILayout::getParticles "virtual SafePointerVector<const IParticle> ILayout::getParticles() const =0

Returns information on all particles (type and abundance) and generates new particles if an  IAbstractParticle denotes a collection 
";

%feature("docstring")  ILayout::getAbundanceOfParticle "virtual double ILayout::getAbundanceOfParticle(size_t index) const =0

Get abundance fraction of particle with index. 
";

%feature("docstring")  ILayout::getTotalAbundance "double ILayout::getTotalAbundance() const

Get total abundance of all particles. 
";

%feature("docstring")  ILayout::getInterferenceFunction "virtual const IInterferenceFunction* ILayout::getInterferenceFunction() const =0

Returns interference function. 
";

%feature("docstring")  ILayout::getTotalParticleSurfaceDensity "virtual double ILayout::getTotalParticleSurfaceDensity() const =0

Returns surface density of all particles. 
";

%feature("docstring")  ILayout::setTotalParticleSurfaceDensity "virtual void ILayout::setTotalParticleSurfaceDensity(double particle_density)=0

Sets surface density of all particles. 
";

%feature("docstring")  ILayout::getApproximation "EInterferenceApproximation ILayout::getApproximation() const

Gets the used approximation for particles and interference functions. 
";

%feature("docstring")  ILayout::setApproximation "void ILayout::setApproximation(EInterferenceApproximation approximation)

Sets the used approximation for particles and interference functions. 
";


// File: classIMaterial.xml
%feature("docstring") IMaterial "

Interface to a named material.

C++ includes: IMaterial.h
";

%feature("docstring")  IMaterial::IMaterial "IMaterial::IMaterial(const std::string &name)
";

%feature("docstring")  IMaterial::~IMaterial "virtual IMaterial::~IMaterial()
";

%feature("docstring")  IMaterial::clone "virtual IMaterial* IMaterial::clone() const =0
";

%feature("docstring")  IMaterial::cloneInverted "virtual IMaterial* IMaterial::cloneInverted() const =0
";

%feature("docstring")  IMaterial::isScalarMaterial "virtual bool IMaterial::isScalarMaterial() const

Indicates whether the interaction with the material is scalar. This means that different polarization states will be diffracted equally 
";

%feature("docstring")  IMaterial::isMagneticMaterial "bool IMaterial::isMagneticMaterial() const 
";

%feature("docstring")  IMaterial::getRefractiveIndex "virtual complex_t IMaterial::getRefractiveIndex() const 
";

%feature("docstring")  IMaterial::getScalarSLD "complex_t IMaterial::getScalarSLD(const WavevectorInfo &wavevectors) const

Returns true if *this agrees with other in all parameters. 
";

%feature("docstring")  IMaterial::getScalarFresnel "complex_t IMaterial::getScalarFresnel(const kvector_t k) const

Return the potential term that is used in the one-dimensional Fresnel calculations. 
";

%feature("docstring")  IMaterial::getPolarizedSLD "Eigen::Matrix2cd IMaterial::getPolarizedSLD(const WavevectorInfo &wavevectors) const

Get the scattering matrix (~potential V) from the material. This matrix appears in the full three-dimensional Schroedinger equation. 
";

%feature("docstring")  IMaterial::getPolarizedFresnel "Eigen::Matrix2cd IMaterial::getPolarizedFresnel(const kvector_t k) const

Return the potential term that is used in the one-dimensional Fresnel calculations. 
";

%feature("docstring")  IMaterial::createTransformedMaterial "virtual const IMaterial* IMaterial::createTransformedMaterial(const Transform3D &transform) const =0

Create a new material that is transformed with respect to this one. 
";


// File: classIMultiLayerBuilder.xml
%feature("docstring") IMultiLayerBuilder "

Interface to the class capable to build samples to simulate.

C++ includes: IMultiLayerBuilder.h
";

%feature("docstring")  IMultiLayerBuilder::IMultiLayerBuilder "IMultiLayerBuilder::IMultiLayerBuilder()
";

%feature("docstring")  IMultiLayerBuilder::~IMultiLayerBuilder "virtual IMultiLayerBuilder::~IMultiLayerBuilder()
";

%feature("docstring")  IMultiLayerBuilder::accept "void IMultiLayerBuilder::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IMultiLayerBuilder::buildSample "virtual MultiLayer* IMultiLayerBuilder::buildSample() const =0
";

%feature("docstring")  IMultiLayerBuilder::set_subtest "void IMultiLayerBuilder::set_subtest(const IParameterized *subtest_item)
";

%feature("docstring")  IMultiLayerBuilder::getFormFactor "const IFormFactor * IMultiLayerBuilder::getFormFactor() const 
";

%feature("docstring")  IMultiLayerBuilder::getFTDistribution2D "const IFTDistribution2D * IMultiLayerBuilder::getFTDistribution2D() const 
";

%feature("docstring")  IMultiLayerBuilder::getChildren "std::vector<const INode*> IMultiLayerBuilder::getChildren() const

Returns a vector of children (const). 
";

%feature("docstring")  IMultiLayerBuilder::onChange "virtual void IMultiLayerBuilder::onChange()

Action to be taken in inherited class when a parameter has changed. 
";


// File: classINamed.xml
%feature("docstring") INamed "

Interface for named objects.

C++ includes: INamed.h
";

%feature("docstring")  INamed::INamed "INamed::INamed()
";

%feature("docstring")  INamed::INamed "INamed::INamed(const std::string &name)
";

%feature("docstring")  INamed::~INamed "virtual INamed::~INamed()
";

%feature("docstring")  INamed::getName "std::string INamed::getName() const 
";


// File: classInfinitePlane.xml
%feature("docstring") InfinitePlane "

The infinite plane is used for masking everything once and forever.

C++ includes: InfinitePlane.h
";

%feature("docstring")  InfinitePlane::InfinitePlane "InfinitePlane::InfinitePlane()
";

%feature("docstring")  InfinitePlane::clone "InfinitePlane* InfinitePlane::clone() const 
";

%feature("docstring")  InfinitePlane::contains "bool InfinitePlane::contains(double, double) const

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  InfinitePlane::contains "bool InfinitePlane::contains(const Bin1D &, const Bin1D &) const

Returns true if area defined by two bins is inside or on border of polygon (more precisely, if mid point of two bins satisfy this condition). 
";


// File: classINode.xml
%feature("docstring") INode "

Base class for tree-like structures containing parameterized objects.

C++ includes: INode.h
";

%feature("docstring")  INode::INode "INode::INode()
";

%feature("docstring")  INode::~INode "virtual INode::~INode()
";

%feature("docstring")  INode::accept "virtual void INode::accept(INodeVisitor *visitor) const =0

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  INode::treeToString "std::string INode::treeToString() const

Returns multiline string representing tree structure below the node. 
";

%feature("docstring")  INode::registerChild "void INode::registerChild(INode *node)
";

%feature("docstring")  INode::getChildren "std::vector< const INode * > INode::getChildren() const

Returns a vector of children (const). 
";

%feature("docstring")  INode::setParent "void INode::setParent(const INode *parent)
";

%feature("docstring")  INode::parent "const INode * INode::parent() const 
";

%feature("docstring")  INode::parent "INode * INode::parent()
";

%feature("docstring")  INode::copyNumber "int INode::copyNumber(const INode *node) const

Returns copyNumber of child, which takes into account existence of children with same name. 
";

%feature("docstring")  INode::displayName "std::string INode::displayName() const

Returns display name, composed from the name of node and it's copy number. 
";

%feature("docstring")  INode::createParameterTree "ParameterPool * INode::createParameterTree() const

Creates new parameter pool, with all local parameters and those of its children. 
";


// File: classINodeVisitor.xml
%feature("docstring") INodeVisitor "

From visitor pattern to achieve double dispatch.

Visitor interface to visit  ISample objects.

C++ includes: INodeVisitor.h
";

%feature("docstring")  INodeVisitor::INodeVisitor "INodeVisitor::INodeVisitor()
";

%feature("docstring")  INodeVisitor::~INodeVisitor "virtual INodeVisitor::~INodeVisitor()
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const INode *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const Beam *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const RectangularDetector *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const SphericalDetector *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IsGISAXSDetector *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ResolutionFunction2DGaussian *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ConvolutionDetectorResolution *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ISample *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IClusteredParticles *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const Crystal *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ILayout *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ParticleLayout *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const Layer *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const LayerInterface *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const MultiLayer *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IAbstractParticle *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IParticle *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const Particle *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ParticleDistribution *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ParticleComposition *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const MesoCrystal *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const ParticleCoreShell *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IFormFactor *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IFormFactorBorn *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IFormFactorDecorator *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorAnisoPyramid *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorBox *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorCone *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorCone6 *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorCuboctahedron *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorCrystal *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorCylinder *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDodecahedron *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorEllipsoidalCylinder *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorFullSphere *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorFullSpheroid *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorGauss *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorHemiEllipsoid *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorIcosahedron *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorLongBoxGauss *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorLongBoxLorentz *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorLorentz *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorPrism3 *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorPrism6 *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorPyramid *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorRipple1 *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorRipple2 *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorSphereGaussianRadius *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorSphereLogNormalRadius *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorTetrahedron *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorTrivial *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorTruncatedCube *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorTruncatedSphere *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorTruncatedSpheroid *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDWBA *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDWBAPol *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorWeighted *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDecoratorDebyeWaller *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDecoratorFactor *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDecoratorMaterial *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDecoratorPositionFactor *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FormFactorDecoratorRotation *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDecayFunction1DCauchy *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDecayFunction1DGauss *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDecayFunction1DTriangle *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDecayFunction1DVoigt *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDecayFunction2DCauchy *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDecayFunction2DGauss *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDecayFunction2DVoigt *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDistribution2DCauchy *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDistribution2DGauss *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDistribution2DGate *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDistribution2DCone *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FTDistribution2DVoigt *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IInterferenceFunction *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const InterferenceFunction1DLattice *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const InterferenceFunctionRadialParaCrystal *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const InterferenceFunction2DLattice *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const InterferenceFunction2DParaCrystal *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const InterferenceFunctionNone *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const LayerRoughness *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IRotation *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const RotationX *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const RotationY *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const RotationZ *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const RotationEuler *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const Instrument *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FitSuiteObjects *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const FitObject *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const GISASSimulation *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const OffSpecSimulation *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IntensityNormalizer *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const IntensityScaleAndShiftNormalizer *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const BasicLattice *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const SquareLattice *)
";

%feature("docstring")  INodeVisitor::visit "virtual void INodeVisitor::visit(const HexagonalLattice *)
";

%feature("docstring")  INodeVisitor::depth "int INodeVisitor::depth() const

Returns depth of the visitor in the composite hierarchy. 
";

%feature("docstring")  INodeVisitor::setDepth "void INodeVisitor::setDepth(int depth)

Sets depth of the visitor in the composite hierarchy. 
";


// File: classINoncopyable.xml
%feature("docstring") INoncopyable "

Mix-in for objects that must not be copied.

This virtual base class disables the copy constructor and the operator= in all its child classes.

C++ includes: INoncopyable.h
";

%feature("docstring")  INoncopyable::INoncopyable "INoncopyable::INoncopyable()
";

%feature("docstring")  INoncopyable::~INoncopyable "virtual INoncopyable::~INoncopyable()
";

%feature("docstring")  INoncopyable::INoncopyable "INoncopyable::INoncopyable(const INoncopyable &)=delete
";


// File: classInstrument.xml
%feature("docstring") Instrument "

Assembles beam, detector and their relative positions wrt the sample.

C++ includes: Instrument.h
";

%feature("docstring")  Instrument::Instrument "Instrument::Instrument()
";

%feature("docstring")  Instrument::Instrument "Instrument::Instrument(const Instrument &other)
";

%feature("docstring")  Instrument::~Instrument "Instrument::~Instrument()
";

%feature("docstring")  Instrument::accept "void Instrument::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  Instrument::getBeam "Beam& Instrument::getBeam()
";

%feature("docstring")  Instrument::getBeam "const Beam& Instrument::getBeam() const 
";

%feature("docstring")  Instrument::setBeam "void Instrument::setBeam(const Beam &beam)
";

%feature("docstring")  Instrument::setBeamParameters "void Instrument::setBeamParameters(double wavelength, double alpha_i, double phi_i)

Sets the beam wavelength and incoming angles. 
";

%feature("docstring")  Instrument::setBeamIntensity "void Instrument::setBeamIntensity(double intensity)

Sets the beam's intensity. 
";

%feature("docstring")  Instrument::setBeamPolarization "void Instrument::setBeamPolarization(const kvector_t bloch_vector)

Sets the beam's polarization according to the given Bloch vector. 
";

%feature("docstring")  Instrument::getBeamIntensity "double Instrument::getBeamIntensity() const

Returns the beam's intensity. 
";

%feature("docstring")  Instrument::getDetector "const IDetector2D * Instrument::getDetector() const

Returns the detector data. 
";

%feature("docstring")  Instrument::getDetector "IDetector2D * Instrument::getDetector()
";

%feature("docstring")  Instrument::getDetectorMask "const DetectorMask * Instrument::getDetectorMask() const 
";

%feature("docstring")  Instrument::getDetectorAxis "const IAxis & Instrument::getDetectorAxis(size_t index) const

Returns a detector axis. 
";

%feature("docstring")  Instrument::getDetectorDimension "size_t Instrument::getDetectorDimension() const

Returns the detector's dimension. 
";

%feature("docstring")  Instrument::setDetector "void Instrument::setDetector(const IDetector2D &detector)

Sets the detector (axes can be overwritten later) 
";

%feature("docstring")  Instrument::setDetectorParameters "void Instrument::setDetectorParameters(size_t n_x, double x_min, double x_max, size_t n_y, double y_min, double y_max)

Sets detector parameters using angle ranges. 
";

%feature("docstring")  Instrument::setDetectorAxes "void Instrument::setDetectorAxes(const IAxis &axis0, const IAxis &axis1)

Sets detector parameters using axes. 
";

%feature("docstring")  Instrument::setDetectorResolutionFunction "void Instrument::setDetectorResolutionFunction(const IResolutionFunction2D &p_resolution_function)

Sets detector resolution function. 
";

%feature("docstring")  Instrument::removeDetectorResolution "void Instrument::removeDetectorResolution()

Removes detector resolution function. 
";

%feature("docstring")  Instrument::setAnalyzerProperties "void Instrument::setAnalyzerProperties(const kvector_t direction, double efficiency, double total_transmission)

Sets the polarization analyzer characteristics of the detector. 
";

%feature("docstring")  Instrument::applyDetectorResolution "void Instrument::applyDetectorResolution(OutputData< double > *p_intensity_map) const

apply the detector resolution to the given intensity map 
";

%feature("docstring")  Instrument::createDetectorIntensity "OutputData< double > * Instrument::createDetectorIntensity(const std::vector< SimulationElement > &elements, IDetector2D::EAxesUnits units=IDetector2D::DEFAULT) const

Returns new intensity map with detector resolution applied and axes in requested units. 
";

%feature("docstring")  Instrument::createDetectorMap "OutputData< double > * Instrument::createDetectorMap(IDetector2D::EAxesUnits units=IDetector2D::DEFAULT) const

Returns empty detector map in given axes units. 
";

%feature("docstring")  Instrument::createSimulationElements "std::vector< SimulationElement > Instrument::createSimulationElements()

Create a vector of  SimulationElement objects according to the beam, detector and its mask. 
";

%feature("docstring")  Instrument::initDetector "void Instrument::initDetector()

init detector with beam settings 
";

%feature("docstring")  Instrument::getChildren "std::vector< const INode * > Instrument::getChildren() const

Returns a vector of children (const). 
";


// File: classIntegratorComplex.xml
%feature("docstring") IntegratorComplex "

Template class to integrate complex class member functions.

Wraps an two integrators from the GNU Scientific Library. Since this class holds a persistent workspace, we need at least one instance per thread. Standard usage for integration inside a class T:
Create a handle to an integrator: 'auto integrator = make_integrator_complex(this, mem_function)'

Call: 'integrator.integrate(lmin, lmax)'

C++ includes: IntegratorComplex.h
";

%feature("docstring")  IntegratorComplex::IntegratorComplex "IntegratorComplex< T >::IntegratorComplex(const T *p_object, complex_integrand< T > p_member_function)

to integrate p_member_function, which must belong to p_object 
";

%feature("docstring")  IntegratorComplex::~IntegratorComplex "IntegratorComplex< T >::~IntegratorComplex()
";

%feature("docstring")  IntegratorComplex::integrate "complex_t IntegratorComplex< T >::integrate(double lmin, double lmax)

perform the actual integration over the range [lmin, lmax] 
";


// File: classIntegratorMCMiser.xml
%feature("docstring") IntegratorMCMiser "

Template class to use Monte Carlo MISER integration of class member functions.

Wraps an integrator from GNU Scientific Library. Since this class holds a persistent workspace, we need at least one instance per thread. Standard usage for integration inside a class T:
Create a handle to an integrator: 'auto integrator = make_integrator_miser(this, mem_function, dimension)'

Call: 'integrator.integrate(lmin, lmax, data, nbr_points)'

C++ includes: IntegratorMCMiser.h
";

%feature("docstring")  IntegratorMCMiser::IntegratorMCMiser "IntegratorMCMiser< T >::IntegratorMCMiser(const T *p_object, miser_integrand< T > p_member_function, size_t dim)

to integrate p_member_function, which must belong to p_object 
";

%feature("docstring")  IntegratorMCMiser::~IntegratorMCMiser "IntegratorMCMiser< T >::~IntegratorMCMiser()
";

%feature("docstring")  IntegratorMCMiser::integrate "double IntegratorMCMiser< T >::integrate(double *min_array, double *max_array, void *params, size_t nbr_points)

perform the actual integration over the ranges [min_array, max_array] 
";


// File: classIntegratorReal.xml
%feature("docstring") IntegratorReal "

Template class to integrate class member functions.

Wraps an integrator from the GNU Scientific Library. Since this class holds a persistent workspace, we need at least one instance per thread. Standard usage for integration inside a class T:
Create a handle to an integrator: 'auto integrator = make_integrator_real(this, mem_function)'

Call: 'integrator.integrate(lmin, lmax)'

C++ includes: IntegratorReal.h
";

%feature("docstring")  IntegratorReal::IntegratorReal "IntegratorReal< T >::IntegratorReal(const T *p_object, real_integrand< T > p_member_function)

to integrate p_member_function, which must belong to p_object 
";

%feature("docstring")  IntegratorReal::~IntegratorReal "IntegratorReal< T >::~IntegratorReal()
";

%feature("docstring")  IntegratorReal::integrate "double IntegratorReal< T >::integrate(double lmin, double lmax)

perform the actual integration over the range [lmin, lmax] 
";


// File: classIntensityDataIOFactory.xml
%feature("docstring") IntensityDataIOFactory "

Provides users with possibility to read and write IntensityData from/to files in different format. Type of the file will be deduced from file name. *.txt - ASCII file with 2D array [nrow][ncol], layout as in numpy. *.int -  BornAgain internal ASCII format. *.tif - 32-bits tiff file. If file name ends woth \"*.gz\" or \"*.bz2\" the file will be zipped on the fly using appropriate algorithm. Usage:

C++ includes: IntensityDataIOFactory.h
";


// File: classIntensityFunctionLog.xml
%feature("docstring") IntensityFunctionLog "

Algorithm for applying log function to the measured intensity.

C++ includes: IIntensityFunction.h
";

%feature("docstring")  IntensityFunctionLog::~IntensityFunctionLog "virtual IntensityFunctionLog::~IntensityFunctionLog()
";

%feature("docstring")  IntensityFunctionLog::clone "virtual IntensityFunctionLog* IntensityFunctionLog::clone() const 
";

%feature("docstring")  IntensityFunctionLog::evaluate "double IntensityFunctionLog::evaluate(double value) const 
";


// File: classIntensityFunctionSqrt.xml
%feature("docstring") IntensityFunctionSqrt "

Algorithm for applying sqrt function to the measured intensity.

C++ includes: IIntensityFunction.h
";

%feature("docstring")  IntensityFunctionSqrt::~IntensityFunctionSqrt "virtual IntensityFunctionSqrt::~IntensityFunctionSqrt()
";

%feature("docstring")  IntensityFunctionSqrt::clone "virtual IntensityFunctionSqrt* IntensityFunctionSqrt::clone() const 
";

%feature("docstring")  IntensityFunctionSqrt::evaluate "double IntensityFunctionSqrt::evaluate(double value) const 
";


// File: classIntensityNormalizer.xml
%feature("docstring") IntensityNormalizer "

Standard  OutputData normalizer, with configurable max_intensity.

C++ includes: IIntensityNormalizer.h
";

%feature("docstring")  IntensityNormalizer::IntensityNormalizer "IntensityNormalizer::IntensityNormalizer(double scale=1.0, double shift=0.0)
";

%feature("docstring")  IntensityNormalizer::~IntensityNormalizer "virtual IntensityNormalizer::~IntensityNormalizer()
";

%feature("docstring")  IntensityNormalizer::clone "IntensityNormalizer * IntensityNormalizer::clone() const 
";

%feature("docstring")  IntensityNormalizer::accept "void IntensityNormalizer::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IntensityNormalizer::createNormalizedData "OutputData< double > * IntensityNormalizer::createNormalizedData(const OutputData< double > &data) const 
";

%feature("docstring")  IntensityNormalizer::apply "void IntensityNormalizer::apply(OutputData< double > &data) const final
";

%feature("docstring")  IntensityNormalizer::setMaximumIntensity "virtual void IntensityNormalizer::setMaximumIntensity(double max_intensity)
";


// File: classIntensityScaleAndShiftNormalizer.xml
%feature("docstring") IntensityScaleAndShiftNormalizer "

Simplified  OutputData normalizer, with max_intensity=1.

C++ includes: IIntensityNormalizer.h
";

%feature("docstring")  IntensityScaleAndShiftNormalizer::IntensityScaleAndShiftNormalizer "IntensityScaleAndShiftNormalizer::IntensityScaleAndShiftNormalizer(double scale=1.0, double shift=0.0)
";

%feature("docstring")  IntensityScaleAndShiftNormalizer::~IntensityScaleAndShiftNormalizer "IntensityScaleAndShiftNormalizer::~IntensityScaleAndShiftNormalizer() final
";

%feature("docstring")  IntensityScaleAndShiftNormalizer::accept "void IntensityScaleAndShiftNormalizer::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IntensityScaleAndShiftNormalizer::setMaximumIntensity "void IntensityScaleAndShiftNormalizer::setMaximumIntensity(double) final
";

%feature("docstring")  IntensityScaleAndShiftNormalizer::clone "IntensityScaleAndShiftNormalizer* IntensityScaleAndShiftNormalizer::clone() const final
";


// File: classInterferenceFunction1DLattice.xml
%feature("docstring") InterferenceFunction1DLattice "

Interference function of 1D lattice.

C++ includes: InterferenceFunction1DLattice.h
";

%feature("docstring")  InterferenceFunction1DLattice::InterferenceFunction1DLattice "InterferenceFunction1DLattice::InterferenceFunction1DLattice(double length, double xi)

Parameters:
-----------

length: 
 Lattice length

xi: 
rotation of lattice with respect to x-axis 
";

%feature("docstring")  InterferenceFunction1DLattice::~InterferenceFunction1DLattice "InterferenceFunction1DLattice::~InterferenceFunction1DLattice() final
";

%feature("docstring")  InterferenceFunction1DLattice::clone "InterferenceFunction1DLattice * InterferenceFunction1DLattice::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  InterferenceFunction1DLattice::accept "void InterferenceFunction1DLattice::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  InterferenceFunction1DLattice::setDecayFunction "void InterferenceFunction1DLattice::setDecayFunction(const IFTDecayFunction1D &pdf)
";

%feature("docstring")  InterferenceFunction1DLattice::getLatticeParameters "Lattice1DParameters InterferenceFunction1DLattice::getLatticeParameters() const 
";

%feature("docstring")  InterferenceFunction1DLattice::getDecayFunction "const IFTDecayFunction1D* InterferenceFunction1DLattice::getDecayFunction() const 
";

%feature("docstring")  InterferenceFunction1DLattice::evaluate "double InterferenceFunction1DLattice::evaluate(const kvector_t q) const final

Evaluates the interference function for a given wavevector transfer (only the real x and y components are relevant) 
";


// File: classInterferenceFunction2DLattice.xml
%feature("docstring") InterferenceFunction2DLattice "

Interference function of 2D lattice.

C++ includes: InterferenceFunction2DLattice.h
";

%feature("docstring")  InterferenceFunction2DLattice::InterferenceFunction2DLattice "InterferenceFunction2DLattice::InterferenceFunction2DLattice(const Lattice2D &lattice)
";

%feature("docstring")  InterferenceFunction2DLattice::InterferenceFunction2DLattice "InterferenceFunction2DLattice::InterferenceFunction2DLattice(double length_1, double length_2, double angle, double xi=0.0)

Parameters:
-----------

length_1: 
 Lattice length 1

length_2: 
 Lattice length 2

angle: 
angle between lattice vectors

xi: 
rotation of lattice with respect to x-axis 
";

%feature("docstring")  InterferenceFunction2DLattice::~InterferenceFunction2DLattice "InterferenceFunction2DLattice::~InterferenceFunction2DLattice() final
";

%feature("docstring")  InterferenceFunction2DLattice::clone "InterferenceFunction2DLattice * InterferenceFunction2DLattice::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  InterferenceFunction2DLattice::accept "void InterferenceFunction2DLattice::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  InterferenceFunction2DLattice::setDecayFunction "void InterferenceFunction2DLattice::setDecayFunction(const IFTDecayFunction2D &pdf)
";

%feature("docstring")  InterferenceFunction2DLattice::getDecayFunction "const IFTDecayFunction2D* InterferenceFunction2DLattice::getDecayFunction() const 
";

%feature("docstring")  InterferenceFunction2DLattice::evaluate "double InterferenceFunction2DLattice::evaluate(const kvector_t q) const final

Evaluates the interference function for a given wavevector transfer (only the real x and y components are relevant) 
";

%feature("docstring")  InterferenceFunction2DLattice::lattice "const Lattice2D & InterferenceFunction2DLattice::lattice() const 
";

%feature("docstring")  InterferenceFunction2DLattice::getParticleDensity "double InterferenceFunction2DLattice::getParticleDensity() const final

Returns the particle density associated with this 2d lattice. 
";

%feature("docstring")  InterferenceFunction2DLattice::getChildren "std::vector< const INode * > InterferenceFunction2DLattice::getChildren() const override

Returns a vector of children (const). 
";

%feature("docstring")  InterferenceFunction2DLattice::onChange "void InterferenceFunction2DLattice::onChange() overridefinal

Action to be taken in inherited class when a parameter has changed. 
";


// File: classInterferenceFunction2DParaCrystal.xml
%feature("docstring") InterferenceFunction2DParaCrystal "

Interference function of 2D paracrystal.

C++ includes: InterferenceFunction2DParaCrystal.h
";

%feature("docstring")  InterferenceFunction2DParaCrystal::InterferenceFunction2DParaCrystal "InterferenceFunction2DParaCrystal::InterferenceFunction2DParaCrystal(const Lattice2D &lattice, double damping_length=0.0, double domain_size_1=0.0, double domain_size_2=0.0)
";

%feature("docstring")  InterferenceFunction2DParaCrystal::InterferenceFunction2DParaCrystal "InterferenceFunction2DParaCrystal::InterferenceFunction2DParaCrystal(double length_1, double length_2, double alpha_lattice, double xi=0.0, double damping_length=0.0)

Parameters:
-----------

length_1: 
Length of first lattice basis vector.

length_2: 
Length of second lattice basis vector.

alpha_lattice: 
Angle between the lattice basis vectors.

xi: 
Angle between first basis vector and the x-axis of incoming beam.

damping_length: 
Damping length for removing delta function singularity at q=0. 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::~InterferenceFunction2DParaCrystal "InterferenceFunction2DParaCrystal::~InterferenceFunction2DParaCrystal() final
";

%feature("docstring")  InterferenceFunction2DParaCrystal::clone "InterferenceFunction2DParaCrystal * InterferenceFunction2DParaCrystal::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::accept "void InterferenceFunction2DParaCrystal::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::setDomainSizes "void InterferenceFunction2DParaCrystal::setDomainSizes(double size_1, double size_2)

Sets the sizes of coherence domains.

Parameters:
-----------

size_1: 
size in first lattice direction

size_2: 
size in second lattice direction 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::setProbabilityDistributions "void InterferenceFunction2DParaCrystal::setProbabilityDistributions(const IFTDistribution2D &pdf_1, const IFTDistribution2D &pdf_2)

Sets the probability distributions (Fourier transformed) for the two lattice directions.

Parameters:
-----------

pdf_1: 
probability distribution in first lattice direction

pdf_2: 
probability distribution in second lattice direction 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::setDampingLength "void InterferenceFunction2DParaCrystal::setDampingLength(double damping_length)
";

%feature("docstring")  InterferenceFunction2DParaCrystal::evaluate "double InterferenceFunction2DParaCrystal::evaluate(const kvector_t q) const final

Evaluates the interference function for a given wavevector transfer (only the real x and y components are relevant) 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::getDomainSizes "std::vector< double > InterferenceFunction2DParaCrystal::getDomainSizes() const 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::getProbabilityDistributions "std::vector< const IFTDistribution2D * > InterferenceFunction2DParaCrystal::getProbabilityDistributions() const 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::setIntegrationOverXi "void InterferenceFunction2DParaCrystal::setIntegrationOverXi(bool integrate_xi)
";

%feature("docstring")  InterferenceFunction2DParaCrystal::getIntegrationOverXi "bool InterferenceFunction2DParaCrystal::getIntegrationOverXi() const 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::getDampingLength "double InterferenceFunction2DParaCrystal::getDampingLength() const 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::lattice "const Lattice2D & InterferenceFunction2DParaCrystal::lattice() const 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::getParticleDensity "double InterferenceFunction2DParaCrystal::getParticleDensity() const final

If defined by this interference function's parameters, returns the particle density (per area). Otherwise, returns zero or a user-defined value 
";

%feature("docstring")  InterferenceFunction2DParaCrystal::getChildren "std::vector< const INode * > InterferenceFunction2DParaCrystal::getChildren() const

Returns a vector of children (const). 
";


// File: classInterferenceFunctionNone.xml
%feature("docstring") InterferenceFunctionNone "

Default interference function (i.e. absence of any interference).

C++ includes: InterferenceFunctionNone.h
";

%feature("docstring")  InterferenceFunctionNone::InterferenceFunctionNone "InterferenceFunctionNone::InterferenceFunctionNone()
";

%feature("docstring")  InterferenceFunctionNone::clone "InterferenceFunctionNone* InterferenceFunctionNone::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  InterferenceFunctionNone::accept "void InterferenceFunctionNone::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  InterferenceFunctionNone::evaluate "double InterferenceFunctionNone::evaluate(const kvector_t) const final

Evaluates the interference function for a given wavevector transfer (only the real x and y components are relevant) 
";


// File: classInterferenceFunctionRadialParaCrystal.xml
%feature("docstring") InterferenceFunctionRadialParaCrystal "

Interference function of radial paracrystal.

C++ includes: InterferenceFunctionRadialParaCrystal.h
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::InterferenceFunctionRadialParaCrystal "InterferenceFunctionRadialParaCrystal::InterferenceFunctionRadialParaCrystal(double peak_distance, double damping_length=0.0)
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::clone "InterferenceFunctionRadialParaCrystal * InterferenceFunctionRadialParaCrystal::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::accept "void InterferenceFunctionRadialParaCrystal::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::setKappa "void InterferenceFunctionRadialParaCrystal::setKappa(double kappa)
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::getKappa "double InterferenceFunctionRadialParaCrystal::getKappa() const final

Retrieves the size-distance coupling constant (default 0.0) 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::setDomainSize "void InterferenceFunctionRadialParaCrystal::setDomainSize(double size)
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::getDomainSize "double InterferenceFunctionRadialParaCrystal::getDomainSize() const 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::evaluate "double InterferenceFunctionRadialParaCrystal::evaluate(const kvector_t q) const final

Evaluates the interference function for a given wavevector transfer (only the real x and y components are relevant) 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::FTPDF "complex_t InterferenceFunctionRadialParaCrystal::FTPDF(double qpar) const 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::setProbabilityDistribution "void InterferenceFunctionRadialParaCrystal::setProbabilityDistribution(const IFTDistribution1D &pdf)
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::getProbabilityDistribution "const IFTDistribution1D* InterferenceFunctionRadialParaCrystal::getProbabilityDistribution() const 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::getPeakDistance "double InterferenceFunctionRadialParaCrystal::getPeakDistance() const 
";

%feature("docstring")  InterferenceFunctionRadialParaCrystal::getDampingLength "double InterferenceFunctionRadialParaCrystal::getDampingLength() const 
";


// File: classIObservable.xml
%feature("docstring") IObservable "

Observable interface from Observer pattern, for 1:n object dependencies.

C++ includes: IObserver.h
";

%feature("docstring")  IObservable::~IObservable "virtual IObservable::~IObservable()
";

%feature("docstring")  IObservable::attachObserver "void IObservable::attachObserver(observer_t obj)

attach observer to the list of observers 
";

%feature("docstring")  IObservable::notifyObservers "void IObservable::notifyObservers()

notify observers about change in status 
";


// File: classIObserver.xml
%feature("docstring") IObserver "

Observer interface from Observer pattern, for 1:n object dependencies.

C++ includes: IObserver.h
";

%feature("docstring")  IObserver::~IObserver "virtual IObserver::~IObserver()
";

%feature("docstring")  IObserver::notify "virtual void IObserver::notify(IObservable *subject)=0

method which is used by observable subject to notify change in status 
";


// File: classIOutputDataReadStrategy.xml
%feature("docstring") IOutputDataReadStrategy "

Interface for reading strategy of  OutputData from file.

C++ includes: OutputDataReadStrategy.h
";

%feature("docstring")  IOutputDataReadStrategy::~IOutputDataReadStrategy "virtual IOutputDataReadStrategy::~IOutputDataReadStrategy()
";

%feature("docstring")  IOutputDataReadStrategy::readOutputData "virtual OutputData<double>* IOutputDataReadStrategy::readOutputData(std::istream &input_stream)=0
";


// File: classIOutputDataWriteStrategy.xml
%feature("docstring") IOutputDataWriteStrategy "

Strategy interface to write OututData in file

C++ includes: OutputDataWriteStrategy.h
";

%feature("docstring")  IOutputDataWriteStrategy::IOutputDataWriteStrategy "IOutputDataWriteStrategy::IOutputDataWriteStrategy()
";

%feature("docstring")  IOutputDataWriteStrategy::~IOutputDataWriteStrategy "virtual IOutputDataWriteStrategy::~IOutputDataWriteStrategy()
";

%feature("docstring")  IOutputDataWriteStrategy::writeOutputData "virtual void IOutputDataWriteStrategy::writeOutputData(const OutputData< double > &data, std::ostream &output_stream)=0
";


// File: classIParameter.xml
%feature("docstring") IParameter "

Pure virtual base class for parameter wrapper classes  RealParameter, ComponentParameter. Holds a pointer to the wrapped parameter, a name, and a callback function to be called when the parameter is changed. This class is templated on the data type of the wrapped parameter.

C++ includes: IParameter.h
";

%feature("docstring")  IParameter::IParameter "IParameter< T >::IParameter()=delete
";

%feature("docstring")  IParameter::IParameter "IParameter< T >::IParameter(const std::string &name, T *data, const std::string &parent_name, const std::function< void()> &onChange)
";

%feature("docstring")  IParameter::clone "virtual IParameter* IParameter< T >::clone(const std::string &new_name=\"\") const =0
";

%feature("docstring")  IParameter::isNull "virtual bool IParameter< T >::isNull() const

Returns true if wrapped parameter was not initialized with proper real value. 
";

%feature("docstring")  IParameter::getData "T& IParameter< T >::getData() const 
";

%feature("docstring")  IParameter::setData "void IParameter< T >::setData(T &data)
";

%feature("docstring")  IParameter::hasSameData "bool IParameter< T >::hasSameData(const IParameter &other)

Returns true if two parameters are pointing to the same raw data. 
";


// File: classIParameterized.xml
%feature("docstring") IParameterized "

Manages a local parameter pool, and a tree of child pools.

C++ includes: IParameterized.h
";

%feature("docstring")  IParameterized::IParameterized "IParameterized::IParameterized(const std::string &name=\"\")
";

%feature("docstring")  IParameterized::IParameterized "IParameterized::IParameterized(const IParameterized &other)
";

%feature("docstring")  IParameterized::~IParameterized "IParameterized::~IParameterized() override
";

%feature("docstring")  IParameterized::getParameterPool "ParameterPool* IParameterized::getParameterPool() const

Returns pointer to the parameter pool. 
";

%feature("docstring")  IParameterized::createParameterTree "ParameterPool * IParameterized::createParameterTree() const

Creates new parameter pool, with all local parameters and those of its children. 
";

%feature("docstring")  IParameterized::parametersToString "std::string IParameterized::parametersToString() const

Returns multiline string representing available parameters. 
";

%feature("docstring")  IParameterized::registerParameter "RealParameter & IParameterized::registerParameter(const std::string &name, double *parpointer)
";

%feature("docstring")  IParameterized::setParameterValue "void IParameterized::setParameterValue(const std::string &name, double value)
";

%feature("docstring")  IParameterized::getParameter "RealParameter * IParameterized::getParameter(const std::string &name) const

Returns parameter with given 'name'. 
";

%feature("docstring")  IParameterized::onChange "virtual void IParameterized::onChange()

Action to be taken in inherited class when a parameter has changed. 
";


// File: classIParticle.xml
%feature("docstring") IParticle "

Pure virtual base class for  Particle,  ParticleComposition,  ParticleCoreShell,  MesoCrystal. Provides position/rotation and form factor. Abundance is inherited from  IAbstractParticle.

C++ includes: IParticle.h
";

%feature("docstring")  IParticle::~IParticle "virtual IParticle::~IParticle()
";

%feature("docstring")  IParticle::clone "virtual IParticle* IParticle::clone() const =0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IParticle::cloneInvertB "virtual IParticle* IParticle::cloneInvertB() const =0

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  IParticle::accept "virtual void IParticle::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IParticle::createFormFactor "IFormFactor * IParticle::createFormFactor() const

Create a form factor for this particle. 
";

%feature("docstring")  IParticle::createTransformedFormFactor "virtual IFormFactor* IParticle::createTransformedFormFactor(const IRotation *p_rotation, kvector_t translation) const =0

Create a form factor for this particle with an extra transformation. 
";

%feature("docstring")  IParticle::getPosition "kvector_t IParticle::getPosition() const

Returns particle position. 
";

%feature("docstring")  IParticle::setPosition "void IParticle::setPosition(kvector_t position)

Sets particle position. 
";

%feature("docstring")  IParticle::setPosition "void IParticle::setPosition(double x, double y, double z)

Sets particle position. 
";

%feature("docstring")  IParticle::getRotation "const IRotation * IParticle::getRotation() const

Returns rotation object. 
";

%feature("docstring")  IParticle::setRotation "void IParticle::setRotation(const IRotation &rotation)

Sets transformation. 
";

%feature("docstring")  IParticle::applyRotation "void IParticle::applyRotation(const IRotation &rotation)

Applies transformation by composing it with the existing one. 
";

%feature("docstring")  IParticle::applyTranslation "void IParticle::applyTranslation(kvector_t displacement)

Applies extra translation by adding it to the current one. 
";

%feature("docstring")  IParticle::getChildren "std::vector< const INode * > IParticle::getChildren() const

Returns a vector of children (const). 
";


// File: classIPixel.xml
%feature("docstring") IPixel "

Interface for a function that maps [0,1]x[0,1] to the kvectors in a pixel.

C++ includes: IPixel.h
";

%feature("docstring")  IPixel::~IPixel "virtual IPixel::~IPixel()
";

%feature("docstring")  IPixel::clone "virtual IPixel* IPixel::clone() const =0
";

%feature("docstring")  IPixel::createZeroSizePixel "virtual IPixel* IPixel::createZeroSizePixel(double x, double y) const =0
";

%feature("docstring")  IPixel::getK "virtual kvector_t IPixel::getK(double x, double y, double wavelength) const =0
";

%feature("docstring")  IPixel::getIntegrationFactor "virtual double IPixel::getIntegrationFactor(double x, double y) const =0
";

%feature("docstring")  IPixel::getSolidAngle "virtual double IPixel::getSolidAngle() const =0
";


// File: classIResolutionFunction2D.xml
%feature("docstring") IResolutionFunction2D "

Interface providing two-dimensional resolution function.

C++ includes: IResolutionFunction2D.h
";

%feature("docstring")  IResolutionFunction2D::~IResolutionFunction2D "virtual IResolutionFunction2D::~IResolutionFunction2D()
";

%feature("docstring")  IResolutionFunction2D::evaluateCDF "virtual double IResolutionFunction2D::evaluateCDF(double x, double y) const =0
";

%feature("docstring")  IResolutionFunction2D::clone "virtual IResolutionFunction2D* IResolutionFunction2D::clone() const =0
";


// File: classIRotation.xml
%feature("docstring") IRotation "

Pure virtual interface for rotations.

C++ includes: Rotations.h
";

%feature("docstring")  IRotation::~IRotation "virtual IRotation::~IRotation()
";

%feature("docstring")  IRotation::clone "virtual IRotation* IRotation::clone() const =0

Returns a clone of this  ISample object. 
";

%feature("docstring")  IRotation::cloneInvertB "IRotation* IRotation::cloneInvertB() const final

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  IRotation::createInverse "virtual IRotation* IRotation::createInverse() const =0

Returns a new  IRotation object that is the current object's inverse. 
";

%feature("docstring")  IRotation::accept "void IRotation::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  IRotation::getTransform3D "virtual Transform3D IRotation::getTransform3D() const =0

Returns transformation. 
";

%feature("docstring")  IRotation::isIdentity "bool IRotation::isIdentity() const

Returns true if roation matrix is identity matrix (no rotations) 
";


// File: classISample.xml
%feature("docstring") ISample "

Pure virtual base class for sample components and properties related to scattering.

C++ includes: ISample.h
";

%feature("docstring")  ISample::clone "ISample* ISample::clone() const override=0

Returns a clone of this  ISample object. 
";

%feature("docstring")  ISample::cloneInvertB "ISample * ISample::cloneInvertB() const

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  ISample::getMaterial "virtual const IMaterial* ISample::getMaterial() const

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  ISample::getAmbientMaterial "virtual const IMaterial* ISample::getAmbientMaterial() const

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  ISample::containedMaterials "std::vector< const IMaterial * > ISample::containedMaterials() const

Returns set of unique materials contained in this  ISample. 
";

%feature("docstring")  ISample::containedSubclass "std::vector< const T * > ISample::containedSubclass() const

Returns vector of children of type T. 
";


// File: classISampleIteratorStrategy.xml
%feature("docstring") ISampleIteratorStrategy "";

%feature("docstring")  ISampleIteratorStrategy::~ISampleIteratorStrategy "virtual ISampleIteratorStrategy::~ISampleIteratorStrategy()
";

%feature("docstring")  ISampleIteratorStrategy::clone "virtual ISampleIteratorStrategy* ISampleIteratorStrategy::clone() const =0
";

%feature("docstring")  ISampleIteratorStrategy::first "virtual IteratorMemento ISampleIteratorStrategy::first(const INode *p_root)=0
";

%feature("docstring")  ISampleIteratorStrategy::next "virtual void ISampleIteratorStrategy::next(IteratorMemento &iterator_stack) const =0
";

%feature("docstring")  ISampleIteratorStrategy::isDone "virtual bool ISampleIteratorStrategy::isDone(IteratorMemento &iterator_stack) const =0
";


// File: classISelectionRule.xml
%feature("docstring") ISelectionRule "

Pure virtual base class for selection rules.

C++ includes: ISelectionRule.h
";

%feature("docstring")  ISelectionRule::~ISelectionRule "virtual ISelectionRule::~ISelectionRule()
";

%feature("docstring")  ISelectionRule::clone "virtual ISelectionRule* ISelectionRule::clone() const =0
";

%feature("docstring")  ISelectionRule::coordinateSelected "virtual bool ISelectionRule::coordinateSelected(const ivector_t &coordinate) const =0
";


// File: classIsGISAXS08BBuilder.xml
%feature("docstring") IsGISAXS08BBuilder "

Builds sample: 2D paracrystal lattice with isotropic pdfs (IsGISAXS example #8).

C++ includes: ParaCrystalBuilder.h
";

%feature("docstring")  IsGISAXS08BBuilder::IsGISAXS08BBuilder "IsGISAXS08BBuilder::IsGISAXS08BBuilder()
";

%feature("docstring")  IsGISAXS08BBuilder::buildSample "MultiLayer * IsGISAXS08BBuilder::buildSample() const 
";


// File: classIsGISAXSDetector.xml
%feature("docstring") IsGISAXSDetector "

A spherical detector used for validation with IsGISAXS results.

C++ includes: IsGISAXSDetector.h
";

%feature("docstring")  IsGISAXSDetector::IsGISAXSDetector "IsGISAXSDetector::IsGISAXSDetector()
";

%feature("docstring")  IsGISAXSDetector::IsGISAXSDetector "IsGISAXSDetector::IsGISAXSDetector(size_t n_phi, double phi_min, double phi_max, size_t n_alpha, double alpha_min, double alpha_max)
";

%feature("docstring")  IsGISAXSDetector::IsGISAXSDetector "IsGISAXSDetector::IsGISAXSDetector(const IsGISAXSDetector &other)
";

%feature("docstring")  IsGISAXSDetector::clone "IsGISAXSDetector * IsGISAXSDetector::clone() const override
";

%feature("docstring")  IsGISAXSDetector::accept "void IsGISAXSDetector::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";


// File: classIShape2D.xml
%feature("docstring") IShape2D "

Basic class for all shapes in 2D.

C++ includes: IShape2D.h
";

%feature("docstring")  IShape2D::IShape2D "IShape2D::IShape2D(const std::string &name)
";

%feature("docstring")  IShape2D::clone "virtual IShape2D* IShape2D::clone() const =0
";

%feature("docstring")  IShape2D::contains "virtual bool IShape2D::contains(double x, double y) const =0

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  IShape2D::contains "virtual bool IShape2D::contains(const Bin1D &binx, const Bin1D &biny) const =0

Returns true if area defined by two bins is inside or on border of polygon (more precisely, if mid point of two bins satisfy this condition). 
";


// File: classISingleton.xml
%feature("docstring") ISingleton "

Base class for singletons.

C++ includes: ISingleton.h
";


// File: classISquaredFunction.xml
%feature("docstring") ISquaredFunction "

Interface providing measures for deviation between two values. Used By  ChiSquaredModule for chi2 calculations.

C++ includes: ISquaredFunction.h
";

%feature("docstring")  ISquaredFunction::ISquaredFunction "ISquaredFunction::ISquaredFunction()
";

%feature("docstring")  ISquaredFunction::~ISquaredFunction "virtual ISquaredFunction::~ISquaredFunction()
";

%feature("docstring")  ISquaredFunction::clone "virtual ISquaredFunction* ISquaredFunction::clone() const =0
";

%feature("docstring")  ISquaredFunction::calculateSquaredDifference "virtual double ISquaredFunction::calculateSquaredDifference(double real_value, double simulated_value) const =0
";

%feature("docstring")  ISquaredFunction::calculateSquaredError "virtual double ISquaredFunction::calculateSquaredError(double real_value, double simulated_value=0.0) const =0
";


// File: classIteratorMemento.xml
%feature("docstring") IteratorMemento "

Holds all iterator states encountered for  SampleTreeIterator.

C++ includes: SampleTreeIterator.h
";

%feature("docstring")  IteratorMemento::IteratorMemento "IteratorMemento::IteratorMemento()
";

%feature("docstring")  IteratorMemento::~IteratorMemento "virtual IteratorMemento::~IteratorMemento()
";

%feature("docstring")  IteratorMemento::push_state "void IteratorMemento::push_state(const IteratorState &state)
";

%feature("docstring")  IteratorMemento::pop_state "void IteratorMemento::pop_state()
";

%feature("docstring")  IteratorMemento::get_state "IteratorState& IteratorMemento::get_state()
";

%feature("docstring")  IteratorMemento::empty "bool IteratorMemento::empty() const 
";

%feature("docstring")  IteratorMemento::reset "void IteratorMemento::reset()
";

%feature("docstring")  IteratorMemento::getCurrent "const INode* IteratorMemento::getCurrent()
";

%feature("docstring")  IteratorMemento::next "void IteratorMemento::next()
";

%feature("docstring")  IteratorMemento::size "size_t IteratorMemento::size() const 
";


// File: classIteratorState.xml
%feature("docstring") IteratorState "

Holds state of iterator at single level for  SampleTreeIterator.

C++ includes: SampleTreeIterator.h
";

%feature("docstring")  IteratorState::IteratorState "IteratorState::IteratorState(const INode *single_element)
";

%feature("docstring")  IteratorState::IteratorState "IteratorState::IteratorState(std::vector< const INode * > samples)
";

%feature("docstring")  IteratorState::~IteratorState "virtual IteratorState::~IteratorState()
";

%feature("docstring")  IteratorState::getCurrent "const INode* IteratorState::getCurrent() const 
";

%feature("docstring")  IteratorState::isEnd "bool IteratorState::isEnd() const 
";

%feature("docstring")  IteratorState::next "void IteratorState::next()
";


// File: classLabelMap.xml
%feature("docstring") LabelMap "";

%feature("docstring")  LabelMap::LabelMap "LabelMap< Key >::LabelMap()
";

%feature("docstring")  LabelMap::~LabelMap "virtual LabelMap< Key >::~LabelMap()
";

%feature("docstring")  LabelMap::clear "void LabelMap< Key >::clear()
";

%feature("docstring")  LabelMap::begin "const_iterator LabelMap< Key >::begin() const 
";

%feature("docstring")  LabelMap::end "const_iterator LabelMap< Key >::end() const 
";

%feature("docstring")  LabelMap::begin "iterator LabelMap< Key >::begin()
";

%feature("docstring")  LabelMap::end "iterator LabelMap< Key >::end()
";

%feature("docstring")  LabelMap::size "size_t LabelMap< Key >::size()
";

%feature("docstring")  LabelMap::insert "void LabelMap< Key >::insert(const Key &key, const std::string &object)
";

%feature("docstring")  LabelMap::find "const iterator LabelMap< Key >::find(const Key &key) const 
";

%feature("docstring")  LabelMap::find "iterator LabelMap< Key >::find(const Key &key)
";

%feature("docstring")  LabelMap::erase "void LabelMap< Key >::erase(const Key &key)
";

%feature("docstring")  LabelMap::value "const std::string& LabelMap< Key >::value(const Key &key)
";


// File: classLargeCylindersInDWBABuilder.xml
%feature("docstring") LargeCylindersInDWBABuilder "

Builds sample with large cylinders for MC integration tests.

C++ includes: CylindersBuilder.h
";

%feature("docstring")  LargeCylindersInDWBABuilder::LargeCylindersInDWBABuilder "LargeCylindersInDWBABuilder::LargeCylindersInDWBABuilder()
";

%feature("docstring")  LargeCylindersInDWBABuilder::buildSample "MultiLayer * LargeCylindersInDWBABuilder::buildSample() const 
";


// File: classLattice.xml
%feature("docstring") Lattice "

A lattice with three basis vectors.

C++ includes: Lattice.h
";

%feature("docstring")  Lattice::Lattice "Lattice::Lattice()=delete
";

%feature("docstring")  Lattice::Lattice "Lattice::Lattice(const kvector_t a1, const kvector_t a2, const kvector_t a3)
";

%feature("docstring")  Lattice::Lattice "Lattice::Lattice(const Lattice &lattice)
";

%feature("docstring")  Lattice::~Lattice "Lattice::~Lattice()
";

%feature("docstring")  Lattice::createTransformedLattice "Lattice Lattice::createTransformedLattice(const Transform3D &transform) const

Create transformed lattice. 
";

%feature("docstring")  Lattice::initialize "void Lattice::initialize() const

Initializes cached data. 
";

%feature("docstring")  Lattice::getBasisVectorA "kvector_t Lattice::getBasisVectorA() const

Returns basis vector a. 
";

%feature("docstring")  Lattice::getBasisVectorB "kvector_t Lattice::getBasisVectorB() const

Returns basis vector b. 
";

%feature("docstring")  Lattice::getBasisVectorC "kvector_t Lattice::getBasisVectorC() const

Returns basis vector c. 
";

%feature("docstring")  Lattice::getVolume "double Lattice::getVolume() const

Returns the volume of the unit cell. 
";

%feature("docstring")  Lattice::getReciprocalLatticeBasis "void Lattice::getReciprocalLatticeBasis(kvector_t b1, kvector_t b2, kvector_t b3) const

Returns the reciprocal basis vectors. 
";

%feature("docstring")  Lattice::getNearestLatticeVectorCoordinates "ivector_t Lattice::getNearestLatticeVectorCoordinates(const kvector_t vector_in) const

Returns the nearest lattice point from a given vector. 
";

%feature("docstring")  Lattice::getNearestReciprocalLatticeVectorCoordinates "ivector_t Lattice::getNearestReciprocalLatticeVectorCoordinates(const kvector_t vector_in) const

Returns the nearest reciprocal lattice point from a given vector. 
";

%feature("docstring")  Lattice::reciprocalLatticeVectorsWithinRadius "std::vector< kvector_t > Lattice::reciprocalLatticeVectorsWithinRadius(const kvector_t input_vector, double radius) const

Computes a list of reciprocal lattice vectors within a specified distance of a given vector. 
";

%feature("docstring")  Lattice::setSelectionRule "void Lattice::setSelectionRule(const ISelectionRule &p_selection_rule)

Sets a selection rule for the reciprocal vectors. 
";


// File: classLattice1DBuilder.xml
%feature("docstring") Lattice1DBuilder "

Builds sample: cylinders with 1DDL structure factor.

C++ includes: LatticeBuilder.h
";

%feature("docstring")  Lattice1DBuilder::Lattice1DBuilder "Lattice1DBuilder::Lattice1DBuilder()
";

%feature("docstring")  Lattice1DBuilder::buildSample "MultiLayer * Lattice1DBuilder::buildSample() const 
";


// File: classLattice1DParameters.xml
%feature("docstring") Lattice1DParameters "

Basic parameters of a one-dimensional lattice.

C++ includes: Lattice1DParameters.h
";

%feature("docstring")  Lattice1DParameters::Lattice1DParameters "Lattice1DParameters::Lattice1DParameters()
";

%feature("docstring")  Lattice1DParameters::Lattice1DParameters "Lattice1DParameters::Lattice1DParameters(double length, double xi)

Parameters:
-----------

length: 
 Lattice constant.

xi: 
TODO: seems unused; explain or remove 
";


// File: classLattice2D.xml
%feature("docstring") Lattice2D "";

%feature("docstring")  Lattice2D::Lattice2D "Lattice2D::Lattice2D(double length1, double length2, double angle, double rotation_angle=0.0)
";

%feature("docstring")  Lattice2D::clone "Lattice2D* Lattice2D::clone() const =0
";

%feature("docstring")  Lattice2D::length1 "double Lattice2D::length1() const 
";

%feature("docstring")  Lattice2D::length2 "virtual double Lattice2D::length2() const 
";

%feature("docstring")  Lattice2D::latticeAngle "double Lattice2D::latticeAngle() const 
";

%feature("docstring")  Lattice2D::rotationAngle "double Lattice2D::rotationAngle() const 
";

%feature("docstring")  Lattice2D::unitCellArea "double Lattice2D::unitCellArea() const 
";

%feature("docstring")  Lattice2D::reciprocalBases "Lattice2D::ReciprocalBases Lattice2D::reciprocalBases() const 
";


// File: classLayer.xml
%feature("docstring") Layer "

A layer, with thickness (in nanometer) and material.

C++ includes: Layer.h
";

%feature("docstring")  Layer::Layer "Layer::Layer(const IMaterial &material, double thickness=0)
";

%feature("docstring")  Layer::~Layer "Layer::~Layer() final
";

%feature("docstring")  Layer::clone "Layer* Layer::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  Layer::cloneInvertB "Layer * Layer::cloneInvertB() const final

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  Layer::accept "void Layer::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  Layer::setThickness "void Layer::setThickness(double thickness)

Sets layer thickness in nanometers. 
";

%feature("docstring")  Layer::getThickness "double Layer::getThickness() const 
";

%feature("docstring")  Layer::setMaterial "void Layer::setMaterial(const IMaterial &material)

Sets  material of the layer. 
";

%feature("docstring")  Layer::getMaterial "const IMaterial* Layer::getMaterial() const

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  Layer::getRefractiveIndex "complex_t Layer::getRefractiveIndex() const 
";

%feature("docstring")  Layer::getRefractiveIndex2 "complex_t Layer::getRefractiveIndex2() const

squared refractive index 
";

%feature("docstring")  Layer::addLayout "void Layer::addLayout(const ILayout &decoration)
";

%feature("docstring")  Layer::getNumberOfLayouts "size_t Layer::getNumberOfLayouts() const 
";

%feature("docstring")  Layer::getLayout "const ILayout * Layer::getLayout(size_t i) const 
";

%feature("docstring")  Layer::hasComputation "bool Layer::hasComputation() const

Returns true if decoration is present. 
";

%feature("docstring")  Layer::getChildren "std::vector< const INode * > Layer::getChildren() const

Returns a vector of children (const). 
";


// File: classLayerInterface.xml
%feature("docstring") LayerInterface "

Interface between two layers, possibly with roughness.

C++ includes: LayerInterface.h
";

%feature("docstring")  LayerInterface::~LayerInterface "LayerInterface::~LayerInterface()
";

%feature("docstring")  LayerInterface::clone "LayerInterface * LayerInterface::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  LayerInterface::accept "virtual void LayerInterface::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  LayerInterface::setRoughness "void LayerInterface::setRoughness(const LayerRoughness &roughness)

Sets roughness of the interface. 
";

%feature("docstring")  LayerInterface::getRoughness "const LayerRoughness * LayerInterface::getRoughness() const

Returns roughness of the interface. 
";

%feature("docstring")  LayerInterface::topLayer "const Layer * LayerInterface::topLayer() const 
";

%feature("docstring")  LayerInterface::bottomLayer "const Layer * LayerInterface::bottomLayer() const 
";

%feature("docstring")  LayerInterface::getChildren "std::vector< const INode * > LayerInterface::getChildren() const

Returns a vector of children (const). 
";


// File: classLayerRoughness.xml
%feature("docstring") LayerRoughness "

A Roughness of interface between two layers.

Based on the article D.K.G. de Boer, Physical review B, Volume 51, Number 8, 15 February 1995 \"X-ray reflection and transmission by rough surfaces\"

C++ includes: LayerRoughness.h
";

%feature("docstring")  LayerRoughness::LayerRoughness "LayerRoughness::LayerRoughness()
";

%feature("docstring")  LayerRoughness::LayerRoughness "LayerRoughness::LayerRoughness(double sigma, double hurstParameter, double lateralCorrLength)
";

%feature("docstring")  LayerRoughness::clone "LayerRoughness* LayerRoughness::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  LayerRoughness::accept "virtual void LayerRoughness::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  LayerRoughness::getSpectralFun "double LayerRoughness::getSpectralFun(const kvector_t kvec) const

Returns power spectral density of the surface roughness.

Power spectral density of the surface roughness is a result of two-dimensional Fourier transform of the correlation function of the roughness profile.

Based on the article D.K.G. de Boer, Physical review B, Volume 51, Number 8, 15 February 1995 \"X-ray reflection and transmission by rough surfaces\" 
";

%feature("docstring")  LayerRoughness::getCorrFun "double LayerRoughness::getCorrFun(const kvector_t k) const

Correlation function of the roughness profile. 
";

%feature("docstring")  LayerRoughness::setSigma "void LayerRoughness::setSigma(double sigma)

Sets rms of roughness. 
";

%feature("docstring")  LayerRoughness::getSigma "double LayerRoughness::getSigma() const

Returns rms of roughness. 
";

%feature("docstring")  LayerRoughness::setHurstParameter "void LayerRoughness::setHurstParameter(double hurstParameter)

Sets hurst parameter. It describes how jagged the surface is. 
";

%feature("docstring")  LayerRoughness::getHurstParameter "double LayerRoughness::getHurstParameter() const

Returns hurst parameter. 
";

%feature("docstring")  LayerRoughness::setLatteralCorrLength "void LayerRoughness::setLatteralCorrLength(double lateralCorrLength)

Sets lateral correlation length. 
";

%feature("docstring")  LayerRoughness::getLatteralCorrLength "double LayerRoughness::getLatteralCorrLength() const

Returns lateral correlation length. 
";


// File: classLayerStrategyBuilder.xml
%feature("docstring") LayerStrategyBuilder "

Methods to generate a simulation strategy for a  ParticleLayoutComputation.

C++ includes: LayerStrategyBuilder.h
";

%feature("docstring")  LayerStrategyBuilder::LayerStrategyBuilder "LayerStrategyBuilder::LayerStrategyBuilder(const MultiLayer *p_multilayer, const ILayout *p_layout, const FullFresnelMap *p_full_map, bool polarized, const SimulationOptions &sim_params, size_t layer_index)
";

%feature("docstring")  LayerStrategyBuilder::~LayerStrategyBuilder "LayerStrategyBuilder::~LayerStrategyBuilder()
";

%feature("docstring")  LayerStrategyBuilder::createStrategy "IInterferenceFunctionStrategy * LayerStrategyBuilder::createStrategy() const

Returns a new strategy object that is able to calculate the scattering for fixed k_f. 
";


// File: classLayersWithAbsorptionBuilder.xml
%feature("docstring") LayersWithAbsorptionBuilder "

The  LayersWithAbsorptionBuilder class generates a multilayer with 3 layers with absorption (refractive index has imaginary part).The middle layer is populated with particles. Requires IComponentService which generates form factors, used for bulk form factors testing.

C++ includes: LayersWithAbsorptionBuilder.h
";

%feature("docstring")  LayersWithAbsorptionBuilder::LayersWithAbsorptionBuilder "LayersWithAbsorptionBuilder::LayersWithAbsorptionBuilder()
";

%feature("docstring")  LayersWithAbsorptionBuilder::~LayersWithAbsorptionBuilder "virtual LayersWithAbsorptionBuilder::~LayersWithAbsorptionBuilder()
";

%feature("docstring")  LayersWithAbsorptionBuilder::buildSample "MultiLayer * LayersWithAbsorptionBuilder::buildSample() const 
";


// File: classLine.xml
%feature("docstring") Line "

A line segment.

C++ includes: Line.h
";

%feature("docstring")  Line::Line "Line::Line(double x1, double y1, double x2, double y2)
";

%feature("docstring")  Line::clone "Line* Line::clone() const 
";

%feature("docstring")  Line::contains "bool Line::contains(double x, double y) const

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  Line::contains "bool Line::contains(const Bin1D &binx, const Bin1D &biny) const

Returns true if area defined by two bins is inside or on border of polygon (more precisely, if mid point of two bins satisfy this condition). 
";


// File: classLLData.xml
%feature("docstring") LLData "

Template class to store data of any type in multi-dimensional space (low-level).

C++ includes: LLData.h
";

%feature("docstring")  LLData::LLData "LLData< T >::LLData(size_t rank, const int *dimensions)
";

%feature("docstring")  LLData::LLData "LLData< T >::LLData(const LLData< T > &right)
";

%feature("docstring")  LLData::~LLData "LLData< T >::~LLData()
";

%feature("docstring")  LLData::meanValues "LLData<double> LLData< T >::meanValues() const 
";

%feature("docstring")  LLData::atCoordinate "T & LLData< T >::atCoordinate(int *coordinate)
";

%feature("docstring")  LLData::atCoordinate "const T & LLData< T >::atCoordinate(int *coordinate) const 
";

%feature("docstring")  LLData::setAll "void LLData< T >::setAll(const T &value)
";

%feature("docstring")  LLData::scaleAll "void LLData< T >::scaleAll(const T &factor)
";

%feature("docstring")  LLData::getTotalSize "size_t LLData< T >::getTotalSize() const 
";

%feature("docstring")  LLData::getRank "size_t LLData< T >::getRank() const 
";

%feature("docstring")  LLData::getDimensions "const int* LLData< T >::getDimensions() const 
";

%feature("docstring")  LLData::getTotalSum "T LLData< T >::getTotalSum() const 
";


// File: classExceptions_1_1LogicErrorException.xml
%feature("docstring") Exceptions::LogicErrorException "";

%feature("docstring")  Exceptions::LogicErrorException::LogicErrorException "Exceptions::LogicErrorException::LogicErrorException(const std::string &message)
";


// File: classMagneticCylindersBuilder.xml
%feature("docstring") MagneticCylindersBuilder "

Builds sample: cylinders with magnetic material and non-zero magnetic field.

C++ includes: MagneticParticlesBuilder.h
";

%feature("docstring")  MagneticCylindersBuilder::MagneticCylindersBuilder "MagneticCylindersBuilder::MagneticCylindersBuilder()
";

%feature("docstring")  MagneticCylindersBuilder::buildSample "MultiLayer * MagneticCylindersBuilder::buildSample() const 
";


// File: classMagneticParticleZeroFieldBuilder.xml
%feature("docstring") MagneticParticleZeroFieldBuilder "

Builds sample: cylinders with magnetic material and zero magnetic field.

C++ includes: MagneticParticlesBuilder.h
";

%feature("docstring")  MagneticParticleZeroFieldBuilder::MagneticParticleZeroFieldBuilder "MagneticParticleZeroFieldBuilder::MagneticParticleZeroFieldBuilder()
";

%feature("docstring")  MagneticParticleZeroFieldBuilder::buildSample "MultiLayer * MagneticParticleZeroFieldBuilder::buildSample() const 
";


// File: classMagneticSubstrateZeroFieldBuilder.xml
%feature("docstring") MagneticSubstrateZeroFieldBuilder "

Builds sample: spheres in substrate layer with a zero magnetic field.

C++ includes: MagneticLayersBuilder.h
";

%feature("docstring")  MagneticSubstrateZeroFieldBuilder::MagneticSubstrateZeroFieldBuilder "MagneticSubstrateZeroFieldBuilder::MagneticSubstrateZeroFieldBuilder()
";

%feature("docstring")  MagneticSubstrateZeroFieldBuilder::buildSample "MultiLayer * MagneticSubstrateZeroFieldBuilder::buildSample() const 
";


// File: classMainComputation.xml
%feature("docstring") MainComputation "

Performs a single-threaded DWBA computation with given sample and simulation parameters, for a given span of detector bins.

Controlled by the multi-threading machinery in  Simulation::runSingleSimulation().

C++ includes: MainComputation.h
";

%feature("docstring")  MainComputation::MainComputation "MainComputation::MainComputation(const MultiLayer &p_multi_layer, const SimulationOptions &options, ProgressHandler &progress, const std::vector< SimulationElement >::iterator &begin_it, const std::vector< SimulationElement >::iterator &end_it)
";

%feature("docstring")  MainComputation::~MainComputation "MainComputation::~MainComputation()
";

%feature("docstring")  MainComputation::run "void MainComputation::run()
";

%feature("docstring")  MainComputation::isCompleted "bool MainComputation::isCompleted() const 
";

%feature("docstring")  MainComputation::errorMessage "std::string MainComputation::errorMessage() const 
";


// File: classMatrixRTCoefficients.xml
%feature("docstring") MatrixRTCoefficients "

Specular reflection and transmission coefficients in a layer in case of 2x2 matrix interactions between the layers and the scattered particle.

C++ includes: MatrixRTCoefficients.h
";

%feature("docstring")  MatrixRTCoefficients::MatrixRTCoefficients "MatrixRTCoefficients::MatrixRTCoefficients()
";

%feature("docstring")  MatrixRTCoefficients::~MatrixRTCoefficients "virtual MatrixRTCoefficients::~MatrixRTCoefficients()
";

%feature("docstring")  MatrixRTCoefficients::clone "MatrixRTCoefficients * MatrixRTCoefficients::clone() const 
";

%feature("docstring")  MatrixRTCoefficients::T1plus "Eigen::Vector2cd MatrixRTCoefficients::T1plus() const

The following functions return the transmitted and reflected amplitudes for different incoming beam polarizations and eigenmodes 
";

%feature("docstring")  MatrixRTCoefficients::R1plus "Eigen::Vector2cd MatrixRTCoefficients::R1plus() const 
";

%feature("docstring")  MatrixRTCoefficients::T2plus "Eigen::Vector2cd MatrixRTCoefficients::T2plus() const 
";

%feature("docstring")  MatrixRTCoefficients::R2plus "Eigen::Vector2cd MatrixRTCoefficients::R2plus() const 
";

%feature("docstring")  MatrixRTCoefficients::T1min "Eigen::Vector2cd MatrixRTCoefficients::T1min() const 
";

%feature("docstring")  MatrixRTCoefficients::R1min "Eigen::Vector2cd MatrixRTCoefficients::R1min() const 
";

%feature("docstring")  MatrixRTCoefficients::T2min "Eigen::Vector2cd MatrixRTCoefficients::T2min() const 
";

%feature("docstring")  MatrixRTCoefficients::R2min "Eigen::Vector2cd MatrixRTCoefficients::R2min() const 
";

%feature("docstring")  MatrixRTCoefficients::getKz "virtual Eigen::Vector2cd MatrixRTCoefficients::getKz() const

Returns z-part of the two wavevector eigenmodes. 
";


// File: classMatrixSpecularInfoMap.xml
%feature("docstring") MatrixSpecularInfoMap "

Implementation of ISpecularInfoMap for matrix valued reflection/transmission coefficients.

C++ includes: MatrixSpecularInfoMap.h
";

%feature("docstring")  MatrixSpecularInfoMap::MatrixSpecularInfoMap "MatrixSpecularInfoMap::MatrixSpecularInfoMap(const MultiLayer *p_multilayer, const MultiLayer *p_inverted_multilayer, size_t layer_index)
";

%feature("docstring")  MatrixSpecularInfoMap::~MatrixSpecularInfoMap "MatrixSpecularInfoMap::~MatrixSpecularInfoMap() final
";

%feature("docstring")  MatrixSpecularInfoMap::clone "MatrixSpecularInfoMap * MatrixSpecularInfoMap::clone() const finaloverride
";

%feature("docstring")  MatrixSpecularInfoMap::getOutCoefficients "const ILayerRTCoefficients * MatrixSpecularInfoMap::getOutCoefficients(const SimulationElement &sim_element) const finaloverride

Retrieves the amplitude coefficients for the given angles. 
";

%feature("docstring")  MatrixSpecularInfoMap::getInCoefficients "const ILayerRTCoefficients * MatrixSpecularInfoMap::getInCoefficients(const SimulationElement &sim_element) const finaloverride

Retrieves the amplitude coefficients for the given angles. 
";


// File: classMesoCrystal.xml
%feature("docstring") MesoCrystal "

A particle with an internal structure of smaller particles.

C++ includes: MesoCrystal.h
";

%feature("docstring")  MesoCrystal::MesoCrystal "MesoCrystal::MesoCrystal(const IClusteredParticles &particle_structure, const IFormFactor &form_factor)
";

%feature("docstring")  MesoCrystal::~MesoCrystal "MesoCrystal::~MesoCrystal()
";

%feature("docstring")  MesoCrystal::clone "MesoCrystal * MesoCrystal::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  MesoCrystal::cloneInvertB "MesoCrystal * MesoCrystal::cloneInvertB() const

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  MesoCrystal::accept "void MesoCrystal::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  MesoCrystal::setAmbientMaterial "void MesoCrystal::setAmbientMaterial(const IMaterial &material)

Sets the refractive index of the ambient material (which influences its scattering power) 
";

%feature("docstring")  MesoCrystal::getAmbientMaterial "const IMaterial * MesoCrystal::getAmbientMaterial() const

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  MesoCrystal::createTransformedFormFactor "IFormFactor * MesoCrystal::createTransformedFormFactor(const IRotation *p_rotation, kvector_t translation) const

Create a form factor for this particle with an extra scattering factor. 
";

%feature("docstring")  MesoCrystal::getClusteredParticles "const IClusteredParticles* MesoCrystal::getClusteredParticles() const

get the internal structure, which is in principle unbounded in space (e.g. an infinite crystal) 
";

%feature("docstring")  MesoCrystal::getChildren "std::vector< const INode * > MesoCrystal::getChildren() const

Returns a vector of children (const). 
";


// File: classMesoCrystalBuilder.xml
%feature("docstring") MesoCrystalBuilder "

Builds sample: mesocrystals of cylindrical shape composed by spherical nanoparticles.

C++ includes: MesoCrystalBuilder.h
";

%feature("docstring")  MesoCrystalBuilder::MesoCrystalBuilder "MesoCrystalBuilder::MesoCrystalBuilder()
";

%feature("docstring")  MesoCrystalBuilder::~MesoCrystalBuilder "virtual MesoCrystalBuilder::~MesoCrystalBuilder()
";

%feature("docstring")  MesoCrystalBuilder::buildSample "MultiLayer * MesoCrystalBuilder::buildSample() const 
";


// File: classMultiLayer.xml
%feature("docstring") MultiLayer "

Our sample model: a stack of layers one below the other.Example of system of 4 layers (3 interfaces):

ambience layer #0 z=getLayerBottomZ(0)=0.0 ------ interface #0 Fe, 20A layer #1 z=getLayerBottomZ(1)=-20.0 ------ interface #1 Cr, 40A layer #2 z=getLayerBottomZ(2)=-60.0 ------ interface #2 substrate layer #3 z=getLayerBottomZ(3)=-60.0

C++ includes: MultiLayer.h
";

%feature("docstring")  MultiLayer::MultiLayer "MultiLayer::MultiLayer()
";

%feature("docstring")  MultiLayer::~MultiLayer "MultiLayer::~MultiLayer()
";

%feature("docstring")  MultiLayer::accept "virtual void MultiLayer::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  MultiLayer::getNumberOfLayers "size_t MultiLayer::getNumberOfLayers() const 
";

%feature("docstring")  MultiLayer::getNumberOfInterfaces "size_t MultiLayer::getNumberOfInterfaces() const 
";

%feature("docstring")  MultiLayer::addLayer "void MultiLayer::addLayer(const Layer &p_child)

Adds object to multilayer, overrides from  ISample.

Adds layer with default (zero) roughness. 
";

%feature("docstring")  MultiLayer::addLayerWithTopRoughness "void MultiLayer::addLayerWithTopRoughness(const Layer &layer, const LayerRoughness &roughness)

Adds layer with top roughness. 
";

%feature("docstring")  MultiLayer::getLayer "const Layer* MultiLayer::getLayer(size_t i_layer) const

Returns layer with given index. 
";

%feature("docstring")  MultiLayer::getLayerInterface "const LayerInterface* MultiLayer::getLayerInterface(size_t i_interface) const

Returns layer with given index. 
";

%feature("docstring")  MultiLayer::getLayerBottomZ "double MultiLayer::getLayerBottomZ(size_t i_layer) const

Returns z-coordinate of the layer's bottom. 
";

%feature("docstring")  MultiLayer::getLayerThickness "double MultiLayer::getLayerThickness(size_t i_layer) const

Returns thickness of layer. 
";

%feature("docstring")  MultiLayer::getLayerTopInterface "const LayerInterface * MultiLayer::getLayerTopInterface(size_t i_layer) const

Returns top interface of layer.

Returns pointer to the top interface of the layer. nInterfaces = nLayers-1, first layer in multilayer doesn't have interface. 
";

%feature("docstring")  MultiLayer::getLayerBottomInterface "const LayerInterface * MultiLayer::getLayerBottomInterface(size_t i_layer) const

Returns bottom interface of layer.

Returns pointer to the bottom interface of the layer. 
";

%feature("docstring")  MultiLayer::addLayout "void MultiLayer::addLayout(const ILayout &layout)

Adds a layout of particles to the whole multilayer (particles can be in different layers) 
";

%feature("docstring")  MultiLayer::getNumberOfLayouts "size_t MultiLayer::getNumberOfLayouts() const 
";

%feature("docstring")  MultiLayer::getLayout "const ILayout * MultiLayer::getLayout(size_t i) const 
";

%feature("docstring")  MultiLayer::clear "void MultiLayer::clear()

Destructs allocated objects. 
";

%feature("docstring")  MultiLayer::clone "MultiLayer * MultiLayer::clone() const

Returns alone of multilayer with clones of all layers and recreated interfaces between layers 
";

%feature("docstring")  MultiLayer::cloneInvertB "MultiLayer * MultiLayer::cloneInvertB() const

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  MultiLayer::setCrossCorrLength "void MultiLayer::setCrossCorrLength(double crossCorrLength)

Sets cross correlation length of roughnesses between interfaces. 
";

%feature("docstring")  MultiLayer::getCrossCorrLength "double MultiLayer::getCrossCorrLength() const

Returns cross correlation length of roughnesses between interfaces. 
";

%feature("docstring")  MultiLayer::getCrossCorrSpectralFun "double MultiLayer::getCrossCorrSpectralFun(const kvector_t kvec, size_t j, size_t k) const

! correlation function of roughnesses between the interfaces

Fourier transform of the correlation function of roughnesses between the interfaces

Fourier transform of the correlation function of roughnesses between the interfaces j,k - indexes of layers in multilayer whose bottom interfaces we are considering 
";

%feature("docstring")  MultiLayer::setLayerThickness "void MultiLayer::setLayerThickness(size_t i_layer, double thickness)

Sets thickness of layer. 
";

%feature("docstring")  MultiLayer::getIndexOfLayer "int MultiLayer::getIndexOfLayer(const Layer *layer) const

returns layer index 
";

%feature("docstring")  MultiLayer::requiresMatrixRTCoefficients "bool MultiLayer::requiresMatrixRTCoefficients() const

returns true if contains magnetic materials and matrix calculations are required 
";

%feature("docstring")  MultiLayer::zToLayerIndex "size_t MultiLayer::zToLayerIndex(double z_value)

returns layer index corresponding to given global z coordinate 
";

%feature("docstring")  MultiLayer::containsMagneticMaterial "bool MultiLayer::containsMagneticMaterial() const 
";

%feature("docstring")  MultiLayer::hasRoughness "bool MultiLayer::hasRoughness() const 
";

%feature("docstring")  MultiLayer::totalNofLayouts "size_t MultiLayer::totalNofLayouts() const 
";

%feature("docstring")  MultiLayer::getChildren "std::vector< const INode * > MultiLayer::getChildren() const

Returns a vector of children (const). 
";


// File: classMultiLayerWithRoughnessBuilder.xml
%feature("docstring") MultiLayerWithRoughnessBuilder "

Builds sample: layers with correlated roughness.

C++ includes: MultiLayerWithRoughnessBuilder.h
";

%feature("docstring")  MultiLayerWithRoughnessBuilder::MultiLayerWithRoughnessBuilder "MultiLayerWithRoughnessBuilder::MultiLayerWithRoughnessBuilder()
";

%feature("docstring")  MultiLayerWithRoughnessBuilder::buildSample "MultiLayer * MultiLayerWithRoughnessBuilder::buildSample() const 
";


// File: classMultipleLayoutBuilder.xml
%feature("docstring") MultipleLayoutBuilder "

Builds sample: mixture of cylinders and prisms without interference, using multiple particle layouts

C++ includes: MultipleLayoutBuilder.h
";

%feature("docstring")  MultipleLayoutBuilder::MultipleLayoutBuilder "MultipleLayoutBuilder::MultipleLayoutBuilder()
";

%feature("docstring")  MultipleLayoutBuilder::buildSample "MultiLayer * MultipleLayoutBuilder::buildSample() const 
";


// File: classExceptions_1_1NotImplementedException.xml
%feature("docstring") Exceptions::NotImplementedException "";

%feature("docstring")  Exceptions::NotImplementedException::NotImplementedException "Exceptions::NotImplementedException::NotImplementedException(const std::string &message)
";


// File: classExceptions_1_1NullPointerException.xml
%feature("docstring") Exceptions::NullPointerException "";

%feature("docstring")  Exceptions::NullPointerException::NullPointerException "Exceptions::NullPointerException::NullPointerException(const std::string &message)
";


// File: classOffSpecSimulation.xml
%feature("docstring") OffSpecSimulation "

Main class to run an off-specular simulation.

C++ includes: OffSpecSimulation.h
";

%feature("docstring")  OffSpecSimulation::OffSpecSimulation "OffSpecSimulation::OffSpecSimulation()
";

%feature("docstring")  OffSpecSimulation::OffSpecSimulation "OffSpecSimulation::OffSpecSimulation(const MultiLayer &p_sample)
";

%feature("docstring")  OffSpecSimulation::OffSpecSimulation "OffSpecSimulation::OffSpecSimulation(const std::shared_ptr< class IMultiLayerBuilder > p_sample_builder)
";

%feature("docstring")  OffSpecSimulation::~OffSpecSimulation "OffSpecSimulation::~OffSpecSimulation() final
";

%feature("docstring")  OffSpecSimulation::clone "OffSpecSimulation* OffSpecSimulation::clone() const 
";

%feature("docstring")  OffSpecSimulation::accept "void OffSpecSimulation::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  OffSpecSimulation::prepareSimulation "void OffSpecSimulation::prepareSimulation() final

Put into a clean state for running a simulation. 
";

%feature("docstring")  OffSpecSimulation::numberOfSimulationElements "int OffSpecSimulation::numberOfSimulationElements() const final

Gets the number of elements this simulation needs to calculate. 
";

%feature("docstring")  OffSpecSimulation::getDetectorIntensity "OutputData<double>* OffSpecSimulation::getDetectorIntensity(IDetector2D::EAxesUnits units_type=IDetector2D::DEFAULT) const

Returns clone of the detector intensity map. 
";

%feature("docstring")  OffSpecSimulation::getIntensityData "Histogram2D * OffSpecSimulation::getIntensityData() const

Returns clone of the detector intensity map in the form of 2D histogram. 
";

%feature("docstring")  OffSpecSimulation::setBeamParameters "void OffSpecSimulation::setBeamParameters(double lambda, const IAxis &alpha_axis, double phi_i)

Sets beam parameters from here (forwarded to  Instrument) 
";

%feature("docstring")  OffSpecSimulation::setDetectorParameters "void OffSpecSimulation::setDetectorParameters(size_t n_x, double x_min, double x_max, size_t n_y, double y_min, double y_max)

Sets detector parameters using angle ranges. 
";


// File: classExceptions_1_1OutOfBoundsException.xml
%feature("docstring") Exceptions::OutOfBoundsException "";

%feature("docstring")  Exceptions::OutOfBoundsException::OutOfBoundsException "Exceptions::OutOfBoundsException::OutOfBoundsException(const std::string &message)
";


// File: classOutputData.xml
%feature("docstring") OutputData "

Template class to store data of any type in multi-dimensional space.Used with data type double to hold simulation results. Used with data type  CumulativeValue in  IHistogram classes. Used with data type bool to hold a detector mask (-> class  DetectorMask)

C++ includes: OutputData.h
";

%feature("docstring")  OutputData::OutputData "OutputData< T >::OutputData()
";

%feature("docstring")  OutputData::OutputData "OutputData< T >::OutputData(const OutputData &)=delete
";

%feature("docstring")  OutputData::~OutputData "OutputData< T >::~OutputData()
";

%feature("docstring")  OutputData::clone "OutputData< T > * OutputData< T >::clone() const 
";

%feature("docstring")  OutputData::copyFrom "void OutputData< T >::copyFrom(const OutputData< T > &x)
";

%feature("docstring")  OutputData::copyShapeFrom "void OutputData< T >::copyShapeFrom(const OutputData< U > &other)
";

%feature("docstring")  OutputData::meanValues "OutputData< double > * OutputData< T >::meanValues() const 
";

%feature("docstring")  OutputData::addAxis "void OutputData< T >::addAxis(const IAxis &new_axis)
";

%feature("docstring")  OutputData::addAxis "void OutputData< T >::addAxis(const std::string &name, size_t size, double start, double end)
";

%feature("docstring")  OutputData::getAxis "const IAxis & OutputData< T >::getAxis(size_t serial_number) const

returns axis with given serial number 
";

%feature("docstring")  OutputData::getAxis "const IAxis & OutputData< T >::getAxis(const std::string &axis_name) const

returns axis with given name 
";

%feature("docstring")  OutputData::getRank "size_t OutputData< T >::getRank() const

Returns number of dimensions. 
";

%feature("docstring")  OutputData::getAllocatedSize "size_t OutputData< T >::getAllocatedSize() const

Returns total size of data buffer (product of bin number in every dimension). 
";

%feature("docstring")  OutputData::getAllSizes "std::vector< size_t > OutputData< T >::getAllSizes() const

Returns all sizes of its axes. 
";

%feature("docstring")  OutputData::getRawDataVector "std::vector< T > OutputData< T >::getRawDataVector() const

Returns copy of raw data vector. 
";

%feature("docstring")  OutputData::fillRawDataArray "void OutputData< T >::fillRawDataArray(T *destination) const

fill raw array with data 
";

%feature("docstring")  OutputData::totalSum "T OutputData< T >::totalSum() const

Returns sum of all values in the data structure. 
";

%feature("docstring")  OutputData::begin "OutputData< T >::iterator OutputData< T >::begin()

Returns read/write iterator that points to the first element. 
";

%feature("docstring")  OutputData::begin "OutputData< T >::const_iterator OutputData< T >::begin() const

Returns read-only iterator that points to the first element. 
";

%feature("docstring")  OutputData::end "iterator OutputData< T >::end()

Returns read/write iterator that points to the one past last element. 
";

%feature("docstring")  OutputData::end "const_iterator OutputData< T >::end() const

Returns read-only iterator that points to the one past last element. 
";

%feature("docstring")  OutputData::getAxesBinIndices "std::vector< int > OutputData< T >::getAxesBinIndices(size_t global_index) const

Returns vector of axes indices for given global index

Parameters:
-----------

global_index: 
The global index of this data structure.

Vector of bin indices for all axes defined 
";

%feature("docstring")  OutputData::getAxisBinIndex "int OutputData< T >::getAxisBinIndex(size_t global_index, size_t i_selected_axis) const

Returns axis bin index for given global index

Parameters:
-----------

global_index: 
The global index of this data structure.

i_selected_axis: 
Serial number of selected axis.

Corresponding bin index for selected axis 
";

%feature("docstring")  OutputData::getAxisBinIndex "int OutputData< T >::getAxisBinIndex(size_t global_index, const std::string &axis_name) const

Returns axis bin index for given global index

Parameters:
-----------

global_index: 
The global index of this data structure.

axis_name: 
The name of selected axis.

Corresponding bin index for selected axis 
";

%feature("docstring")  OutputData::toGlobalIndex "size_t OutputData< T >::toGlobalIndex(const std::vector< int > &axes_indices) const

Returns global index for specified indices of axes

Parameters:
-----------

axes_indices: 
Vector of axes indices for all specified axes in this dataset

Corresponding global index 
";

%feature("docstring")  OutputData::findGlobalIndex "size_t OutputData< T >::findGlobalIndex(const std::vector< double > &coordinates) const

Returns global index for specified axes values

Parameters:
-----------

coordinates: 
Vector of axes coordinates for all specified axes in this dataset

Closest global index 
";

%feature("docstring")  OutputData::getAxisValue "double OutputData< T >::getAxisValue(size_t global_index, size_t i_selected_axis) const

Returns the value of selected axis for given global_index.

Parameters:
-----------

global_index: 
The global index of this data structure.

i_selected_axis: 
Serial number of selected axis.

corresponding bin center of selected axis 
";

%feature("docstring")  OutputData::getAxisValue "double OutputData< T >::getAxisValue(size_t global_index, const std::string &axis_name) const

Returns the value of selected axis for given global_index.

Parameters:
-----------

global_index: 
The global index of this data structure.

axis_name: 
The name of selected axis.

corresponding bin center of selected axis 
";

%feature("docstring")  OutputData::getAxesValues "std::vector< double > OutputData< T >::getAxesValues(size_t global_index) const

Returns values on all defined axes for given globalbin number

Parameters:
-----------

global_index: 
The global index of this data structure.

Vector of corresponding bin centers 
";

%feature("docstring")  OutputData::getAxisBin "Bin1D OutputData< T >::getAxisBin(size_t global_index, size_t i_selected_axis) const

Returns bin of selected axis for given global_index.

Parameters:
-----------

global_index: 
The global index of this data structure.

i_selected_axis: 
Serial number of selected axis.

Corresponding  Bin1D object 
";

%feature("docstring")  OutputData::getAxisBin "Bin1D OutputData< T >::getAxisBin(size_t global_index, const std::string &axis_name) const

Returns bin of selected axis for given global_index.

Parameters:
-----------

global_index: 
The global index of this data structure.

axis_name: 
The name of selected axis.

Corresponding  Bin1D object 
";

%feature("docstring")  OutputData::clear "void OutputData< T >::clear()

Sets object into initial state (no dimensions, data) 
";

%feature("docstring")  OutputData::setAllTo "void OutputData< T >::setAllTo(const T &value)

Sets content of output data to specific value. 
";

%feature("docstring")  OutputData::scaleAll "void OutputData< T >::scaleAll(const T &factor)

multiply every item of this output data by value 
";

%feature("docstring")  OutputData::setAxisSizes "void OutputData< T >::setAxisSizes(size_t rank, int *n_dims)

Adds 'rank' axes with indicated sizes. 
";

%feature("docstring")  OutputData::setRawDataVector "void OutputData< T >::setRawDataVector(const std::vector< T > &data_vector)

Sets new values to raw data vector. 
";

%feature("docstring")  OutputData::setRawDataArray "void OutputData< T >::setRawDataArray(const T *source)

Sets new values to raw data array. 
";

%feature("docstring")  OutputData::getValue "double OutputData< T >::getValue(size_t index) const

Returns value or summed value, depending on T. 
";

%feature("docstring")  OutputData::hasSameDimensions "bool OutputData< T >::hasSameDimensions(const OutputData< U > &right) const

Returns true if object have same dimensions and number of axes bins.

Returns true if object have same dimensions. 
";

%feature("docstring")  OutputData::hasSameShape "bool OutputData< T >::hasSameShape(const OutputData< U > &right) const

Returns true if objects a) have same dimensions b) bin boundaries of axes coincide.

Returns true if object have same dimensions and shape of axis. 
";

%feature("docstring")  OutputData::isInitialized "bool OutputData< T >::isInitialized() const

returns data as Python numpy array

returns true if object is correctly initialized 
";

%feature("docstring")  OutputData::allocate "void OutputData< T >::allocate()

memory allocation for current dimensions configuration 
";


// File: classOutputDataIterator.xml
%feature("docstring") OutputDataIterator "

Iterator for underlying  OutputData container.

C++ includes: OutputDataIterator.h
";

%feature("docstring")  OutputDataIterator::OutputDataIterator "OutputDataIterator< TValue, TContainer >::OutputDataIterator()

Empty constructor to comply with stl forward iterators. 
";

%feature("docstring")  OutputDataIterator::OutputDataIterator "OutputDataIterator< TValue, TContainer >::OutputDataIterator(TContainer *p_output_data, size_t start_at_index=0)

constructor 
";

%feature("docstring")  OutputDataIterator::OutputDataIterator "OutputDataIterator< TValue, TContainer >::OutputDataIterator(const OutputDataIterator< TValue2, TContainer2 > &other)

templated copy construction 
";

%feature("docstring")  OutputDataIterator::OutputDataIterator "OutputDataIterator< TValue, TContainer >::OutputDataIterator(const OutputDataIterator< TValue, TContainer > &other)

non-templated copy construction 
";

%feature("docstring")  OutputDataIterator::~OutputDataIterator "OutputDataIterator< TValue, TContainer >::~OutputDataIterator()
";

%feature("docstring")  OutputDataIterator::getIndex "size_t OutputDataIterator< TValue, TContainer >::getIndex() const

Returns current index. 
";

%feature("docstring")  OutputDataIterator::getContainer "TContainer* OutputDataIterator< TValue, TContainer >::getContainer() const

Returns container pointer. 
";

%feature("docstring")  OutputDataIterator::swap "void OutputDataIterator< TValue, TContainer >::swap(OutputDataIterator< TValue, TContainer > &other)

Swaps iterators. 
";


// File: classOutputDataReader.xml
%feature("docstring") OutputDataReader "

Reads  OutputData from file using different reading strategies.

C++ includes: OutputDataReader.h
";

%feature("docstring")  OutputDataReader::OutputDataReader "OutputDataReader::OutputDataReader(const std::string &file_name)
";

%feature("docstring")  OutputDataReader::getOutputData "OutputData< double > * OutputDataReader::getOutputData()

read output data from file (file name was set already from OutputDataIOFactory) 
";

%feature("docstring")  OutputDataReader::setStrategy "void OutputDataReader::setStrategy(IOutputDataReadStrategy *read_strategy)

Sets concrete reading strategy. 
";


// File: classOutputDataReadFactory.xml
%feature("docstring") OutputDataReadFactory "

Creates reader appropariate for given type of files.

C++ includes: OutputDataReadFactory.h
";


// File: classOutputDataReadINTStrategy.xml
%feature("docstring") OutputDataReadINTStrategy "

Strategy to read  BornAgain native IntensityData from ASCII file.

C++ includes: OutputDataReadStrategy.h
";

%feature("docstring")  OutputDataReadINTStrategy::readOutputData "OutputData< double > * OutputDataReadINTStrategy::readOutputData(std::istream &input_stream)
";


// File: classOutputDataReadNumpyTXTStrategy.xml
%feature("docstring") OutputDataReadNumpyTXTStrategy "

Strategy to read  OutputData from simple ASCII file with the layout as in numpy.savetxt.

C++ includes: OutputDataReadStrategy.h
";

%feature("docstring")  OutputDataReadNumpyTXTStrategy::readOutputData "OutputData< double > * OutputDataReadNumpyTXTStrategy::readOutputData(std::istream &input_stream)
";


// File: classOutputDataWriteFactory.xml
%feature("docstring") OutputDataWriteFactory "

Creates writer appropariate for given type of files.

C++ includes: OutputDataWriteFactory.h
";


// File: classOutputDataWriteINTStrategy.xml
%feature("docstring") OutputDataWriteINTStrategy "

Strategy to write  OutputData to special  BornAgain ASCII format

C++ includes: OutputDataWriteStrategy.h
";

%feature("docstring")  OutputDataWriteINTStrategy::writeOutputData "void OutputDataWriteINTStrategy::writeOutputData(const OutputData< double > &data, std::ostream &output_stream)
";


// File: classOutputDataWriteNumpyTXTStrategy.xml
%feature("docstring") OutputDataWriteNumpyTXTStrategy "

Strategy to write  OutputData to simple ASCII file with the layout as in numpy.savetxt

C++ includes: OutputDataWriteStrategy.h
";

%feature("docstring")  OutputDataWriteNumpyTXTStrategy::writeOutputData "void OutputDataWriteNumpyTXTStrategy::writeOutputData(const OutputData< double > &data, std::ostream &output_stream)
";


// File: classOutputDataWriter.xml
%feature("docstring") OutputDataWriter "

Write  OutputData to file using different witing strategies.

C++ includes: OutputDataWriter.h
";

%feature("docstring")  OutputDataWriter::OutputDataWriter "OutputDataWriter::OutputDataWriter(const std::string &file_name)
";

%feature("docstring")  OutputDataWriter::writeOutputData "void OutputDataWriter::writeOutputData(const OutputData< double > &data)

Writes output data to file. 
";

%feature("docstring")  OutputDataWriter::setStrategy "void OutputDataWriter::setStrategy(IOutputDataWriteStrategy *write_strategy)

Sets concrete writing strategy. 
";


// File: classParameterDistribution.xml
%feature("docstring") ParameterDistribution "";

%feature("docstring")  ParameterDistribution::ParameterDistribution "ParameterDistribution::ParameterDistribution(const std::string &par_name, const IDistribution1D &distribution, size_t nbr_samples, double sigma_factor=0.0, const RealLimits &limits=RealLimits())
";

%feature("docstring")  ParameterDistribution::ParameterDistribution "ParameterDistribution::ParameterDistribution(const std::string &par_name, const IDistribution1D &distribution, size_t nbr_samples, double xmin, double xmax)
";

%feature("docstring")  ParameterDistribution::ParameterDistribution "ParameterDistribution::ParameterDistribution(const ParameterDistribution &other)
";

%feature("docstring")  ParameterDistribution::~ParameterDistribution "ParameterDistribution::~ParameterDistribution()
";

%feature("docstring")  ParameterDistribution::linkParameter "ParameterDistribution & ParameterDistribution::linkParameter(std::string par_name)
";

%feature("docstring")  ParameterDistribution::getMainParameterName "std::string ParameterDistribution::getMainParameterName() const

get the main parameter's name 
";

%feature("docstring")  ParameterDistribution::getNbrSamples "size_t ParameterDistribution::getNbrSamples() const

get number of samples for this distribution 
";

%feature("docstring")  ParameterDistribution::getSigmaFactor "double ParameterDistribution::getSigmaFactor() const

get the sigma factor 
";

%feature("docstring")  ParameterDistribution::getDistribution "const IDistribution1D * ParameterDistribution::getDistribution() const 
";

%feature("docstring")  ParameterDistribution::generateSamples "std::vector< ParameterSample > ParameterDistribution::generateSamples() const

generate list of sampled values with their weight 
";

%feature("docstring")  ParameterDistribution::getLinkedParameterNames "std::vector<std::string> ParameterDistribution::getLinkedParameterNames() const

get list of linked parameter names 
";

%feature("docstring")  ParameterDistribution::getLimits "RealLimits ParameterDistribution::getLimits() const 
";

%feature("docstring")  ParameterDistribution::getMinValue "double ParameterDistribution::getMinValue() const 
";

%feature("docstring")  ParameterDistribution::getMaxValue "double ParameterDistribution::getMaxValue() const 
";


// File: classParameterPattern.xml
%feature("docstring") ParameterPattern "

Helper class for constructing parameter patterns.

C++ includes: ParameterPattern.h
";

%feature("docstring")  ParameterPattern::ParameterPattern "ParameterPattern::ParameterPattern()
";

%feature("docstring")  ParameterPattern::ParameterPattern "ParameterPattern::ParameterPattern(std::string root_object)
";

%feature("docstring")  ParameterPattern::beginsWith "ParameterPattern & ParameterPattern::beginsWith(std::string start_type)
";

%feature("docstring")  ParameterPattern::add "ParameterPattern & ParameterPattern::add(std::string object_type)
";

%feature("docstring")  ParameterPattern::toStdString "std::string ParameterPattern::toStdString() const 
";


// File: classParameterPool.xml
%feature("docstring") ParameterPool "

Container with parameters for  IParameterized object.

C++ includes: ParameterPool.h
";

%feature("docstring")  ParameterPool::ParameterPool "ParameterPool::ParameterPool()

Constructs an empty parameter pool. 
";

%feature("docstring")  ParameterPool::~ParameterPool "ParameterPool::~ParameterPool()
";

%feature("docstring")  ParameterPool::clone "ParameterPool * ParameterPool::clone() const

Returns a literal clone. 
";

%feature("docstring")  ParameterPool::copyToExternalPool "void ParameterPool::copyToExternalPool(const std::string &prefix, ParameterPool *external_pool) const

Copies parameters of given pool to  other pool, prepeding  prefix to the parameter names. 
";

%feature("docstring")  ParameterPool::clear "void ParameterPool::clear()

Clears the parameter map. 
";

%feature("docstring")  ParameterPool::size "size_t ParameterPool::size() const

Returns number of parameters in the pool. 
";

%feature("docstring")  ParameterPool::addParameter "RealParameter & ParameterPool::addParameter(RealParameter *par)

Adds parameter to the pool, and returns reference to the input pointer.

Returning the input pointer allows us to concatenate function calls like pool->addParameter( new  RealParameter(...) ).setLimits(-1,+1).setFixed().setUnit(\"nm\") 
";

%feature("docstring")  ParameterPool::getParameter "RealParameter * ParameterPool::getParameter(const std::string &name)

Returns parameter with given  name. 
";

%feature("docstring")  ParameterPool::getParameter "const RealParameter * ParameterPool::getParameter(const std::string &name) const

Returns parameter with given  name. 
";

%feature("docstring")  ParameterPool::getParameters "const std::vector<RealParameter*> ParameterPool::getParameters() const

Returns full vector of parameters. 
";

%feature("docstring")  ParameterPool::getMatchedParameters "std::vector< RealParameter * > ParameterPool::getMatchedParameters(const std::string &wildcards) const

Returns nonempty vector of parameters that match the  pattern ('*' allowed), or throws. 
";

%feature("docstring")  ParameterPool::getUniqueMatch "RealParameter * ParameterPool::getUniqueMatch(const std::string &pattern) const

Returns the one parameter that matches the  pattern (wildcards '*' allowed), or throws. 
";

%feature("docstring")  ParameterPool::setParameterValue "void ParameterPool::setParameterValue(const std::string &name, double value)

Sets parameter value. 
";

%feature("docstring")  ParameterPool::setMatchedParametersValue "int ParameterPool::setMatchedParametersValue(const std::string &wildcards, double value)

Sets value of the nonzero parameters that match  pattern ('*' allowed), or throws. 
";

%feature("docstring")  ParameterPool::setUniqueMatchValue "void ParameterPool::setUniqueMatchValue(const std::string &pattern, double value)

Sets value of the one parameter that matches  pattern ('*' allowed), or throws. 
";

%feature("docstring")  ParameterPool::parameterNames "std::vector< std::string > ParameterPool::parameterNames() const 
";


// File: classParameterSample.xml
%feature("docstring") ParameterSample "

A parameter value with a weight, as obtained when sampling from a distribution.

C++ includes: ParameterSample.h
";

%feature("docstring")  ParameterSample::ParameterSample "ParameterSample::ParameterSample(double _value=0., double _weight=1.)
";


// File: classParticle.xml
%feature("docstring") Particle "

A particle with a form factor and refractive index.

C++ includes: Particle.h
";

%feature("docstring")  Particle::Particle "Particle::Particle()
";

%feature("docstring")  Particle::Particle "Particle::Particle(const IMaterial &p_material)
";

%feature("docstring")  Particle::Particle "Particle::Particle(const IMaterial &p_material, const IFormFactor &form_factor)
";

%feature("docstring")  Particle::Particle "Particle::Particle(const IMaterial &p_material, const IFormFactor &form_factor, const IRotation &rotation)
";

%feature("docstring")  Particle::clone "Particle * Particle::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  Particle::cloneInvertB "Particle * Particle::cloneInvertB() const

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  Particle::accept "virtual void Particle::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  Particle::setAmbientMaterial "void Particle::setAmbientMaterial(const IMaterial &material) final

Sets the refractive index of the ambient material (which influences its scattering power) 
";

%feature("docstring")  Particle::getAmbientMaterial "const IMaterial* Particle::getAmbientMaterial() const final

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  Particle::createTransformedFormFactor "IFormFactor * Particle::createTransformedFormFactor(const IRotation *p_rotation, kvector_t translation) const

Create a form factor for this particle with an extra scattering factor. 
";

%feature("docstring")  Particle::setMaterial "void Particle::setMaterial(const IMaterial &material)
";

%feature("docstring")  Particle::getMaterial "const IMaterial* Particle::getMaterial() const

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  Particle::getRefractiveIndex "complex_t Particle::getRefractiveIndex() const 
";

%feature("docstring")  Particle::setFormFactor "void Particle::setFormFactor(const IFormFactor &form_factor)
";

%feature("docstring")  Particle::getFormFactor "const IFormFactor* Particle::getFormFactor() const 
";

%feature("docstring")  Particle::getChildren "std::vector< const INode * > Particle::getChildren() const

Returns a vector of children (const). 
";


// File: classParticleComposition.xml
%feature("docstring") ParticleComposition "

A composition of particles at fixed positions

C++ includes: ParticleComposition.h
";

%feature("docstring")  ParticleComposition::ParticleComposition "ParticleComposition::ParticleComposition()
";

%feature("docstring")  ParticleComposition::ParticleComposition "ParticleComposition::ParticleComposition(const IParticle &particle)
";

%feature("docstring")  ParticleComposition::ParticleComposition "ParticleComposition::ParticleComposition(const IParticle &particle, kvector_t position)
";

%feature("docstring")  ParticleComposition::ParticleComposition "ParticleComposition::ParticleComposition(const IParticle &particle, std::vector< kvector_t > positions)
";

%feature("docstring")  ParticleComposition::~ParticleComposition "ParticleComposition::~ParticleComposition()
";

%feature("docstring")  ParticleComposition::clone "ParticleComposition * ParticleComposition::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  ParticleComposition::cloneInvertB "ParticleComposition * ParticleComposition::cloneInvertB() const

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  ParticleComposition::accept "virtual void ParticleComposition::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  ParticleComposition::addParticle "void ParticleComposition::addParticle(const IParticle &particle)
";

%feature("docstring")  ParticleComposition::addParticle "void ParticleComposition::addParticle(const IParticle &particle, kvector_t position)
";

%feature("docstring")  ParticleComposition::addParticles "void ParticleComposition::addParticles(const IParticle &particle, std::vector< kvector_t > positions)
";

%feature("docstring")  ParticleComposition::setAmbientMaterial "void ParticleComposition::setAmbientMaterial(const IMaterial &material)

Sets the refractive index of the ambient material (which influences its scattering power) 
";

%feature("docstring")  ParticleComposition::getAmbientMaterial "const IMaterial * ParticleComposition::getAmbientMaterial() const

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  ParticleComposition::createTransformedFormFactor "IFormFactor * ParticleComposition::createTransformedFormFactor(const IRotation *p_rotation, kvector_t translation) const

Create a form factor for this particle with an extra scattering factor. 
";

%feature("docstring")  ParticleComposition::getNbrParticles "size_t ParticleComposition::getNbrParticles() const

Returns number of different particles. 
";

%feature("docstring")  ParticleComposition::getParticle "const IParticle * ParticleComposition::getParticle(size_t index) const

Returns particle with given index. 
";

%feature("docstring")  ParticleComposition::getParticlePosition "kvector_t ParticleComposition::getParticlePosition(size_t index) const 
";

%feature("docstring")  ParticleComposition::getChildren "std::vector< const INode * > ParticleComposition::getChildren() const

Returns a vector of children (const). 
";


// File: classParticleCompositionBuilder.xml
%feature("docstring") ParticleCompositionBuilder "

Builds sample: two layers of spheres at hex lattice.

C++ includes: ParticleCompositionBuilder.h
";

%feature("docstring")  ParticleCompositionBuilder::ParticleCompositionBuilder "ParticleCompositionBuilder::ParticleCompositionBuilder()
";

%feature("docstring")  ParticleCompositionBuilder::buildSample "MultiLayer * ParticleCompositionBuilder::buildSample() const 
";


// File: classParticleCoreShell.xml
%feature("docstring") ParticleCoreShell "

A particle with a core/shell geometry.

C++ includes: ParticleCoreShell.h
";

%feature("docstring")  ParticleCoreShell::ParticleCoreShell "ParticleCoreShell::ParticleCoreShell(const Particle &shell, const Particle &core, kvector_t relative_core_position=kvector_t(0.0, 0.0, 0.0))
";

%feature("docstring")  ParticleCoreShell::~ParticleCoreShell "ParticleCoreShell::~ParticleCoreShell()
";

%feature("docstring")  ParticleCoreShell::clone "ParticleCoreShell * ParticleCoreShell::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  ParticleCoreShell::cloneInvertB "ParticleCoreShell * ParticleCoreShell::cloneInvertB() const final

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  ParticleCoreShell::accept "void ParticleCoreShell::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  ParticleCoreShell::setAmbientMaterial "void ParticleCoreShell::setAmbientMaterial(const IMaterial &material) final

Sets the refractive index of the ambient material (which influences its scattering power) 
";

%feature("docstring")  ParticleCoreShell::getAmbientMaterial "const IMaterial * ParticleCoreShell::getAmbientMaterial() const final

Returns nullptr, unless overwritten to return a specific material. 
";

%feature("docstring")  ParticleCoreShell::createTransformedFormFactor "IFormFactor * ParticleCoreShell::createTransformedFormFactor(const IRotation *p_rotation, kvector_t translation) const final

Create a form factor for this particle with an extra scattering factor. 
";

%feature("docstring")  ParticleCoreShell::coreParticle "const Particle * ParticleCoreShell::coreParticle() const 
";

%feature("docstring")  ParticleCoreShell::shellParticle "const Particle * ParticleCoreShell::shellParticle() const 
";

%feature("docstring")  ParticleCoreShell::getChildren "std::vector< const INode * > ParticleCoreShell::getChildren() const

Returns a vector of children (const). 
";


// File: classParticleDistribution.xml
%feature("docstring") ParticleDistribution "

A particle type that is a parametric distribution of  IParticle's.

C++ includes: ParticleDistribution.h
";

%feature("docstring")  ParticleDistribution::ParticleDistribution "ParticleDistribution::ParticleDistribution(const IParticle &prototype, const ParameterDistribution &par_distr)
";

%feature("docstring")  ParticleDistribution::clone "ParticleDistribution * ParticleDistribution::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  ParticleDistribution::cloneInvertB "ParticleDistribution * ParticleDistribution::cloneInvertB() const final

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  ParticleDistribution::accept "void ParticleDistribution::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  ParticleDistribution::setAmbientMaterial "void ParticleDistribution::setAmbientMaterial(const IMaterial &material) final

Sets the refractive index of the ambient material. 
";

%feature("docstring")  ParticleDistribution::getAmbientMaterial "const IMaterial * ParticleDistribution::getAmbientMaterial() const final

Returns particle's material. 
";

%feature("docstring")  ParticleDistribution::generateParticles "std::vector< const IParticle * > ParticleDistribution::generateParticles() const

Returns list of new particles generated according to a distribution.

Returns particle clones with parameter values drawn from distribution. 
";

%feature("docstring")  ParticleDistribution::getParameterDistribution "ParameterDistribution ParticleDistribution::getParameterDistribution() const

Returns the distributed parameter data. 
";

%feature("docstring")  ParticleDistribution::getParticle "const IParticle* ParticleDistribution::getParticle() const

Returns particle. 
";

%feature("docstring")  ParticleDistribution::getChildren "std::vector< const INode * > ParticleDistribution::getChildren() const

Returns a vector of children (const). 
";


// File: classParticleInTheAirBuilder.xml
%feature("docstring") ParticleInTheAirBuilder "

The  ParticleInTheAirBuilder class generates a multilayer with single air layer populated with particles of certain types. Requires IComponentService which generates form factors, used for bulk form factors testing.

C++ includes: ParticleInTheAirBuilder.h
";

%feature("docstring")  ParticleInTheAirBuilder::ParticleInTheAirBuilder "ParticleInTheAirBuilder::ParticleInTheAirBuilder()
";

%feature("docstring")  ParticleInTheAirBuilder::~ParticleInTheAirBuilder "virtual ParticleInTheAirBuilder::~ParticleInTheAirBuilder()
";

%feature("docstring")  ParticleInTheAirBuilder::buildSample "MultiLayer * ParticleInTheAirBuilder::buildSample() const 
";


// File: classParticleLayout.xml
%feature("docstring") ParticleLayout "

Decorator class that adds particles to  ISample objects.

C++ includes: ParticleLayout.h
";

%feature("docstring")  ParticleLayout::ParticleLayout "ParticleLayout::ParticleLayout()
";

%feature("docstring")  ParticleLayout::ParticleLayout "ParticleLayout::ParticleLayout(const IAbstractParticle &particle)
";

%feature("docstring")  ParticleLayout::ParticleLayout "ParticleLayout::ParticleLayout(const IAbstractParticle &particle, double abundance)
";

%feature("docstring")  ParticleLayout::~ParticleLayout "ParticleLayout::~ParticleLayout() final
";

%feature("docstring")  ParticleLayout::clone "ParticleLayout * ParticleLayout::clone() const final

Returns a clone of this  ISample object. 
";

%feature("docstring")  ParticleLayout::cloneInvertB "ParticleLayout * ParticleLayout::cloneInvertB() const final

Returns a clone with inverted magnetic fields. 
";

%feature("docstring")  ParticleLayout::accept "void ParticleLayout::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  ParticleLayout::addParticle "void ParticleLayout::addParticle(const IAbstractParticle &particle)

Adds generic particle to the layout. 
";

%feature("docstring")  ParticleLayout::addParticle "void ParticleLayout::addParticle(const IAbstractParticle &particle, double abundance)

Adds generic particle to the layout with only abundance defined.

Parameters:
-----------

particle: 
to be added

abundance: 
 Particle abundance 
";

%feature("docstring")  ParticleLayout::addParticle "void ParticleLayout::addParticle(const IParticle &particle, double abundance, const kvector_t position)

Adds particle to the layout with abundance and position defined.

Parameters:
-----------

particle: 
to be added

abundance: 
 Particle abundance

position: 
 Particle position 
";

%feature("docstring")  ParticleLayout::addParticle "void ParticleLayout::addParticle(const IParticle &particle, double abundance, const kvector_t position, const IRotation &rotation)

Adds particle to the layout with abundance, position and the rotation defined.

Parameters:
-----------

particle: 
to be added

abundance: 
 Particle abundance

position: 
 Particle position

rotation: 
 Particle rotation 
";

%feature("docstring")  ParticleLayout::getNumberOfParticles "size_t ParticleLayout::getNumberOfParticles() const final

Returns number of particles. 
";

%feature("docstring")  ParticleLayout::getParticle "const IAbstractParticle * ParticleLayout::getParticle(size_t index) const final

Returns particle info. 
";

%feature("docstring")  ParticleLayout::getParticles "SafePointerVector< const IParticle > ParticleLayout::getParticles() const final

Returns information on all particles (type and abundance) and generates new particles if an  IAbstractParticle denotes a collection 
";

%feature("docstring")  ParticleLayout::getAbundanceOfParticle "double ParticleLayout::getAbundanceOfParticle(size_t index) const

Returns the abundance fraction of particle at given index. 
";

%feature("docstring")  ParticleLayout::getInterferenceFunction "const IInterferenceFunction * ParticleLayout::getInterferenceFunction() const final

Returns interference function. 
";

%feature("docstring")  ParticleLayout::addInterferenceFunction "void ParticleLayout::addInterferenceFunction(const IInterferenceFunction &interference_function)

Adds interference functions. 
";

%feature("docstring")  ParticleLayout::getTotalParticleSurfaceDensity "double ParticleLayout::getTotalParticleSurfaceDensity() const final

Returns surface density of all particles. 
";

%feature("docstring")  ParticleLayout::setTotalParticleSurfaceDensity "void ParticleLayout::setTotalParticleSurfaceDensity(double particle_density) final

Sets surface density of all particles. 
";

%feature("docstring")  ParticleLayout::getChildren "std::vector< const INode * > ParticleLayout::getChildren() const

Returns a vector of children (const). 
";


// File: classParticleLayoutComputation.xml
%feature("docstring") ParticleLayoutComputation "

Computes the scattering contribution from one particle layout. Controlled by  MainComputation.

C++ includes: ParticleLayoutComputation.h
";

%feature("docstring")  ParticleLayoutComputation::ParticleLayoutComputation "ParticleLayoutComputation::ParticleLayoutComputation(const MultiLayer *p_multilayer, const ILayout *p_layout, size_t layer_index)
";

%feature("docstring")  ParticleLayoutComputation::eval "bool ParticleLayoutComputation::eval(const SimulationOptions &options, ProgressHandler *progress, bool polarized, const std::vector< SimulationElement >::iterator &begin_it, const std::vector< SimulationElement >::iterator &end_it) const override

Computes scattering intensity for given range of simulation elements. 
";


// File: classPolygon.xml
%feature("docstring") Polygon "

A polygon in 2D space.Polygon defined by two arrays with x and y coordinates of points. Sizes of arrays should coincide. If polygon is unclosed (the last point doesn't repeat the first one), it will be closed automatically.

C++ includes: Polygon.h
";

%feature("docstring")  Polygon::Polygon "Polygon::Polygon(const std::vector< double > x, const std::vector< double > y)

Parameters:
-----------

x: 
Vector of x-coordinates of polygon points.

y: 
Vector of y-coordinates of polygon points. 
";

%feature("docstring")  Polygon::Polygon "Polygon::Polygon(const std::vector< std::vector< double >> points)
";

%feature("docstring")  Polygon::Polygon "Polygon::Polygon(const PolygonPrivate *d)
";

%feature("docstring")  Polygon::~Polygon "Polygon::~Polygon()
";

%feature("docstring")  Polygon::clone "virtual Polygon* Polygon::clone() const 
";

%feature("docstring")  Polygon::contains "bool Polygon::contains(double x, double y) const

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  Polygon::contains "bool Polygon::contains(const Bin1D &binx, const Bin1D &biny) const

Returns true if area defined by two bins is inside or on border of polygon (more precisely, if mid point of two bins satisfy this condition). 
";

%feature("docstring")  Polygon::getArea "double Polygon::getArea() const 
";

%feature("docstring")  Polygon::getPoints "void Polygon::getPoints(std::vector< double > &xpos, std::vector< double > &ypos) const 
";


// File: classPolygonalTopology.xml
%feature("docstring") PolygonalTopology "

For internal use in  PolyhedralFace.

C++ includes: FormFactorPolyhedron.h
";


// File: classPolygonPrivate.xml
%feature("docstring") PolygonPrivate "

The private data for polygons to hide boost dependency from the header. 
";

%feature("docstring")  PolygonPrivate::init_from "void PolygonPrivate::init_from(const std::vector< double > &x, const std::vector< double > &y)
";

%feature("docstring")  PolygonPrivate::get_points "void PolygonPrivate::get_points(std::vector< double > &xpos, std::vector< double > &ypos)
";


// File: classPolyhedralEdge.xml
%feature("docstring") PolyhedralEdge "

One edge of a polygon, for form factor computation.

C++ includes: FormFactorPolyhedron.h
";

%feature("docstring")  PolyhedralEdge::PolyhedralEdge "PolyhedralEdge::PolyhedralEdge(const kvector_t _Vlow, const kvector_t _Vhig)
";

%feature("docstring")  PolyhedralEdge::E "kvector_t PolyhedralEdge::E() const 
";

%feature("docstring")  PolyhedralEdge::R "kvector_t PolyhedralEdge::R() const 
";

%feature("docstring")  PolyhedralEdge::qE "complex_t PolyhedralEdge::qE(cvector_t q) const 
";

%feature("docstring")  PolyhedralEdge::qR "complex_t PolyhedralEdge::qR(cvector_t q) const 
";

%feature("docstring")  PolyhedralEdge::contrib "complex_t PolyhedralEdge::contrib(int m, const cvector_t qpa, complex_t qrperp) const

Returns sum_l=0^M/2 u^2l v^(M-2l) / (2l+1)!(M-2l)! - vperp^M/M! 
";


// File: classPolyhedralFace.xml
%feature("docstring") PolyhedralFace "

A polygon, for form factor computation.

C++ includes: FormFactorPolyhedron.h
";

%feature("docstring")  PolyhedralFace::PolyhedralFace "PolyhedralFace::PolyhedralFace(const std::vector< kvector_t > &_V=std::vector< kvector_t >(), bool _sym_S2=false)

Sets internal variables for given vertex chain.

Parameters:
-----------

V: 
oriented vertex list

_sym_S2: 
true if face has a perpedicular two-fold symmetry axis 
";

%feature("docstring")  PolyhedralFace::area "double PolyhedralFace::area() const 
";

%feature("docstring")  PolyhedralFace::center "kvector_t PolyhedralFace::center() const 
";

%feature("docstring")  PolyhedralFace::pyramidalVolume "double PolyhedralFace::pyramidalVolume() const 
";

%feature("docstring")  PolyhedralFace::radius3d "double PolyhedralFace::radius3d() const 
";

%feature("docstring")  PolyhedralFace::normalProjectionConj "complex_t PolyhedralFace::normalProjectionConj(cvector_t q) const

Returns conj(q)*normal [ BasicVector3D::dot is antilinear in 'this' argument]. 
";

%feature("docstring")  PolyhedralFace::ff_n "complex_t PolyhedralFace::ff_n(int m, const cvector_t q) const

Returns contribution qn*f_n [of order q^(n+1)] from this face to the polyhedral form factor. 
";

%feature("docstring")  PolyhedralFace::ff "complex_t PolyhedralFace::ff(const cvector_t q, const bool sym_Ci) const

Returns the contribution ff(q) of this face to the polyhedral form factor. 
";

%feature("docstring")  PolyhedralFace::ff_2D "complex_t PolyhedralFace::ff_2D(const cvector_t qpa) const

Returns the two-dimensional form factor of this face, for use in a prism. 
";

%feature("docstring")  PolyhedralFace::assert_Ci "void PolyhedralFace::assert_Ci(const PolyhedralFace &other) const

Throws if deviation from inversion symmetry is detected. Does not check vertices. 
";


// File: classPolyhedralTopology.xml
%feature("docstring") PolyhedralTopology "

For internal use in  FormFactorPolyhedron.

C++ includes: FormFactorPolyhedron.h
";


// File: classPrecomputed.xml
%feature("docstring") Precomputed "

This class contains precomputed constants.

C++ includes: Precomputed.h
";


// File: classProgressHandler.xml
%feature("docstring") ProgressHandler "

Maintains information about progress of a computation. Owner is the computation, which periodically calls the thread-safe function incrementDone(..). An application (GUI or script) may subscribe(..) to be informed about progress. It is then periodically called back by inform(..). The return value of inform(..) can be used to request termination of the computation.

C++ includes: ProgressHandler.h
";

%feature("docstring")  ProgressHandler::ProgressHandler "ProgressHandler::ProgressHandler()
";

%feature("docstring")  ProgressHandler::ProgressHandler "ProgressHandler::ProgressHandler(const ProgressHandler &other)
";

%feature("docstring")  ProgressHandler::subscribe "void ProgressHandler::subscribe(ProgressHandler::Callback_t callback)
";

%feature("docstring")  ProgressHandler::reset "void ProgressHandler::reset()
";

%feature("docstring")  ProgressHandler::setExpectedNTicks "void ProgressHandler::setExpectedNTicks(size_t n)
";

%feature("docstring")  ProgressHandler::incrementDone "void ProgressHandler::incrementDone(size_t ticks_done)

Increments number of completed computation steps (ticks). Performs callback (method m_inform) to inform the subscriber about the state of the computation and to obtain as return value a flag that indicates whether to continue the computation. 
";

%feature("docstring")  ProgressHandler::alive "bool ProgressHandler::alive()
";


// File: classRadialParaCrystalBuilder.xml
%feature("docstring") RadialParaCrystalBuilder "

Builds sample: cylinders with 1DDL structure factor (IsGISAXS example #4).

C++ includes: ParaCrystalBuilder.h
";

%feature("docstring")  RadialParaCrystalBuilder::RadialParaCrystalBuilder "RadialParaCrystalBuilder::RadialParaCrystalBuilder()
";

%feature("docstring")  RadialParaCrystalBuilder::buildSample "MultiLayer * RadialParaCrystalBuilder::buildSample() const 
";


// File: classRealParameter.xml
%feature("docstring") RealParameter "

Wraps a parameter of type double. In addition to name and on-change callback held by the parent class  IParameter, this class holds Limits, Attributes (currently only fixed or not), and a  Unit.

C++ includes: RealParameter.h
";

%feature("docstring")  RealParameter::RealParameter "RealParameter::RealParameter(const std::string &name, double *par, const std::string &parent_name=std::string(), const std::function< void()> &onChange=std::function< void()>(), const RealLimits &limits=RealLimits::limitless(), const Attributes &attr=Attributes::free())
";

%feature("docstring")  RealParameter::clone "RealParameter * RealParameter::clone(const std::string &new_name=\"\") const 
";

%feature("docstring")  RealParameter::setValue "void RealParameter::setValue(double value)

Sets value of wrapped parameter and emit signal. 
";

%feature("docstring")  RealParameter::getValue "double RealParameter::getValue() const

Returns value of wrapped parameter. 
";

%feature("docstring")  RealParameter::setLimits "RealParameter& RealParameter::setLimits(const RealLimits &limits)
";

%feature("docstring")  RealParameter::getLimits "RealLimits RealParameter::getLimits() const 
";

%feature("docstring")  RealParameter::setLimited "RealParameter & RealParameter::setLimited(double lower, double upper)
";

%feature("docstring")  RealParameter::setPositive "RealParameter & RealParameter::setPositive()
";

%feature("docstring")  RealParameter::setNonnegative "RealParameter & RealParameter::setNonnegative()
";

%feature("docstring")  RealParameter::setUnit "RealParameter& RealParameter::setUnit(const std::string &name)
";

%feature("docstring")  RealParameter::unit "std::string RealParameter::unit() const 
";


// File: structLattice2D_1_1ReciprocalBases.xml
%feature("docstring") Lattice2D::ReciprocalBases "";


// File: classRectangle.xml
%feature("docstring") Rectangle "

The rectangle shape having its axis aligned to the (non-rotated) coordinate system.

C++ includes: Rectangle.h
";

%feature("docstring")  Rectangle::Rectangle "Rectangle::Rectangle(double xlow, double ylow, double xup, double yup)

Parameters:
-----------

xlow: 
x-coordinate of lower left corner

ylow: 
y-coordinate of lower left corner

xup: 
x-coordinate of upper right corner

yup: 
y-coordinate of upper right corner 
";

%feature("docstring")  Rectangle::clone "Rectangle* Rectangle::clone() const 
";

%feature("docstring")  Rectangle::contains "bool Rectangle::contains(double x, double y) const

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  Rectangle::contains "bool Rectangle::contains(const Bin1D &binx, const Bin1D &biny) const

Returns true if area defined by two bins is inside or on border of polygon (more precisely, if mid point of two bins satisfy this condition). 
";

%feature("docstring")  Rectangle::getArea "double Rectangle::getArea() const 
";

%feature("docstring")  Rectangle::getXlow "double Rectangle::getXlow() const 
";

%feature("docstring")  Rectangle::getYlow "double Rectangle::getYlow() const 
";

%feature("docstring")  Rectangle::getXup "double Rectangle::getXup() const 
";

%feature("docstring")  Rectangle::getYup "double Rectangle::getYup() const 
";


// File: classRectangularDetector.xml
%feature("docstring") RectangularDetector "

A flat rectangular detector with axes and resolution function.

C++ includes: RectangularDetector.h
";

%feature("docstring")  RectangularDetector::RectangularDetector "RectangularDetector::RectangularDetector(int nxbins, double width, int nybins, double height)

Rectangular detector constructor

Parameters:
-----------

nxbins: 
Number of bins (pixels) in x-direction

width: 
Width of the detector in mm along x-direction

nybins: 
Number of bins (pixels) in y-direction

height: 
Height of the detector in mm along y-direction 
";

%feature("docstring")  RectangularDetector::RectangularDetector "RectangularDetector::RectangularDetector(const RectangularDetector &other)
";

%feature("docstring")  RectangularDetector::clone "RectangularDetector * RectangularDetector::clone() const override
";

%feature("docstring")  RectangularDetector::accept "void RectangularDetector::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  RectangularDetector::~RectangularDetector "RectangularDetector::~RectangularDetector()
";

%feature("docstring")  RectangularDetector::init "void RectangularDetector::init(const Beam &beam) override

Inits detector with the beam settings. 
";

%feature("docstring")  RectangularDetector::setPosition "void RectangularDetector::setPosition(const kvector_t normal_to_detector, double u0, double v0, const kvector_t direction=kvector_t(0.0,-1.0, 0.0))
";

%feature("docstring")  RectangularDetector::setPerpendicularToSampleX "void RectangularDetector::setPerpendicularToSampleX(double distance, double u0, double v0)
";

%feature("docstring")  RectangularDetector::setPerpendicularToDirectBeam "void RectangularDetector::setPerpendicularToDirectBeam(double distance, double u0, double v0)
";

%feature("docstring")  RectangularDetector::setPerpendicularToReflectedBeam "void RectangularDetector::setPerpendicularToReflectedBeam(double distance, double u0=0.0, double v0=0.0)
";

%feature("docstring")  RectangularDetector::setDirectBeamPosition "void RectangularDetector::setDirectBeamPosition(double u0, double v0)
";

%feature("docstring")  RectangularDetector::getWidth "double RectangularDetector::getWidth() const 
";

%feature("docstring")  RectangularDetector::getHeight "double RectangularDetector::getHeight() const 
";

%feature("docstring")  RectangularDetector::getNbinsX "size_t RectangularDetector::getNbinsX() const 
";

%feature("docstring")  RectangularDetector::getNbinsY "size_t RectangularDetector::getNbinsY() const 
";

%feature("docstring")  RectangularDetector::getNormalVector "kvector_t RectangularDetector::getNormalVector() const 
";

%feature("docstring")  RectangularDetector::getU0 "double RectangularDetector::getU0() const 
";

%feature("docstring")  RectangularDetector::getV0 "double RectangularDetector::getV0() const 
";

%feature("docstring")  RectangularDetector::getDirectionVector "kvector_t RectangularDetector::getDirectionVector() const 
";

%feature("docstring")  RectangularDetector::getDistance "double RectangularDetector::getDistance() const 
";

%feature("docstring")  RectangularDetector::getDirectBeamU0 "double RectangularDetector::getDirectBeamU0() const 
";

%feature("docstring")  RectangularDetector::getDirectBeamV0 "double RectangularDetector::getDirectBeamV0() const 
";

%feature("docstring")  RectangularDetector::getDetectorArrangment "RectangularDetector::EDetectorArrangement RectangularDetector::getDetectorArrangment() const 
";

%feature("docstring")  RectangularDetector::getValidAxesUnits "std::vector< IDetector2D::EAxesUnits > RectangularDetector::getValidAxesUnits() const override

returns vector of valid axes units 
";

%feature("docstring")  RectangularDetector::getDefaultAxesUnits "IDetector2D::EAxesUnits RectangularDetector::getDefaultAxesUnits() const override

return default axes units 
";


// File: classRectangularPixel.xml
%feature("docstring") RectangularPixel "";

%feature("docstring")  RectangularPixel::RectangularPixel "RectangularPixel::RectangularPixel(kvector_t corner_pos, kvector_t width, kvector_t height)
";

%feature("docstring")  RectangularPixel::~RectangularPixel "virtual RectangularPixel::~RectangularPixel()
";

%feature("docstring")  RectangularPixel::clone "RectangularPixel * RectangularPixel::clone() const override
";

%feature("docstring")  RectangularPixel::createZeroSizePixel "RectangularPixel * RectangularPixel::createZeroSizePixel(double x, double y) const override
";

%feature("docstring")  RectangularPixel::getK "kvector_t RectangularPixel::getK(double x, double y, double wavelength) const override
";

%feature("docstring")  RectangularPixel::getIntegrationFactor "double RectangularPixel::getIntegrationFactor(double x, double y) const override
";

%feature("docstring")  RectangularPixel::getSolidAngle "double RectangularPixel::getSolidAngle() const override
";


// File: classRectParaCrystalBuilder.xml
%feature("docstring") RectParaCrystalBuilder "

Builds sample: 2D paracrystal lattice (IsGISAXS example #8).

C++ includes: ParaCrystalBuilder.h
";

%feature("docstring")  RectParaCrystalBuilder::RectParaCrystalBuilder "RectParaCrystalBuilder::RectParaCrystalBuilder()
";

%feature("docstring")  RectParaCrystalBuilder::buildSample "MultiLayer * RectParaCrystalBuilder::buildSample() const 
";


// File: classRegionOfInterest.xml
%feature("docstring") RegionOfInterest "

Defines rectangular area for the detector which will be simulated/fitted.

C++ includes: RegionOfInterest.h
";

%feature("docstring")  RegionOfInterest::RegionOfInterest "RegionOfInterest::RegionOfInterest(const IDetector2D &detector, double xlow, double ylow, double xup, double yup)
";

%feature("docstring")  RegionOfInterest::RegionOfInterest "RegionOfInterest::RegionOfInterest(const OutputData< double > &data, double xlow, double ylow, double xup, double yup)
";

%feature("docstring")  RegionOfInterest::clone "RegionOfInterest * RegionOfInterest::clone() const 
";

%feature("docstring")  RegionOfInterest::~RegionOfInterest "RegionOfInterest::~RegionOfInterest()
";

%feature("docstring")  RegionOfInterest::getXlow "double RegionOfInterest::getXlow() const 
";

%feature("docstring")  RegionOfInterest::getYlow "double RegionOfInterest::getYlow() const 
";

%feature("docstring")  RegionOfInterest::getXup "double RegionOfInterest::getXup() const 
";

%feature("docstring")  RegionOfInterest::getYup "double RegionOfInterest::getYup() const 
";

%feature("docstring")  RegionOfInterest::detectorIndex "size_t RegionOfInterest::detectorIndex(size_t roiIndex) const

Converts roi index to the detector index. 
";

%feature("docstring")  RegionOfInterest::roiIndex "size_t RegionOfInterest::roiIndex(size_t detectorIndex) const

Converts global detector index to ROI index. 
";

%feature("docstring")  RegionOfInterest::roiSize "size_t RegionOfInterest::roiSize() const

Number of detector bins in ROI area. 
";

%feature("docstring")  RegionOfInterest::detectorSize "size_t RegionOfInterest::detectorSize() const

Number of detector bins. 
";

%feature("docstring")  RegionOfInterest::isInROI "bool RegionOfInterest::isInROI(size_t detectorIndex) const 
";

%feature("docstring")  RegionOfInterest::clipAxisToRoi "std::unique_ptr< IAxis > RegionOfInterest::clipAxisToRoi(size_t axis_index, const IAxis &axis) const 
";


// File: classResolutionFunction2DGaussian.xml
%feature("docstring") ResolutionFunction2DGaussian "

Simple gaussian two-dimensional resolution function.

C++ includes: ResolutionFunction2DGaussian.h
";

%feature("docstring")  ResolutionFunction2DGaussian::ResolutionFunction2DGaussian "ResolutionFunction2DGaussian::ResolutionFunction2DGaussian(double sigma_x, double sigma_y)
";

%feature("docstring")  ResolutionFunction2DGaussian::evaluateCDF "double ResolutionFunction2DGaussian::evaluateCDF(double x, double y) const 
";

%feature("docstring")  ResolutionFunction2DGaussian::clone "ResolutionFunction2DGaussian* ResolutionFunction2DGaussian::clone() const 
";

%feature("docstring")  ResolutionFunction2DGaussian::accept "void ResolutionFunction2DGaussian::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  ResolutionFunction2DGaussian::getSigmaX "double ResolutionFunction2DGaussian::getSigmaX() const 
";

%feature("docstring")  ResolutionFunction2DGaussian::getSigmaY "double ResolutionFunction2DGaussian::getSigmaY() const 
";


// File: classRotatedPyramidsBuilder.xml
%feature("docstring") RotatedPyramidsBuilder "

Builds sample: Pyramids, rotated pyramids on top of substrate (IsGISAXS example #9)

C++ includes: RotatedPyramidsBuilder.h
";

%feature("docstring")  RotatedPyramidsBuilder::RotatedPyramidsBuilder "RotatedPyramidsBuilder::RotatedPyramidsBuilder()
";

%feature("docstring")  RotatedPyramidsBuilder::buildSample "MultiLayer * RotatedPyramidsBuilder::buildSample() const 
";


// File: classRotatedSquareLatticeBuilder.xml
%feature("docstring") RotatedSquareLatticeBuilder "

Builds sample: 2D lattice with different disorder (IsGISAXS example #6).

C++ includes: TwoDimLatticeBuilder.h
";

%feature("docstring")  RotatedSquareLatticeBuilder::RotatedSquareLatticeBuilder "RotatedSquareLatticeBuilder::RotatedSquareLatticeBuilder()
";

%feature("docstring")  RotatedSquareLatticeBuilder::buildSample "MultiLayer * RotatedSquareLatticeBuilder::buildSample() const 
";


// File: classRotationEuler.xml
%feature("docstring") RotationEuler "";

%feature("docstring")  RotationEuler::RotationEuler "RotationEuler::RotationEuler(double alpha, double beta, double gamma)
";

%feature("docstring")  RotationEuler::clone "RotationEuler* RotationEuler::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  RotationEuler::createInverse "IRotation * RotationEuler::createInverse() const

Returns a new  IRotation object that is the current object's inverse. 
";

%feature("docstring")  RotationEuler::accept "void RotationEuler::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  RotationEuler::getAlpha "double RotationEuler::getAlpha() const 
";

%feature("docstring")  RotationEuler::getBeta "double RotationEuler::getBeta() const 
";

%feature("docstring")  RotationEuler::getGamma "double RotationEuler::getGamma() const 
";

%feature("docstring")  RotationEuler::getTransform3D "Transform3D RotationEuler::getTransform3D() const

Returns transformation. 
";


// File: classRotationX.xml
%feature("docstring") RotationX "";

%feature("docstring")  RotationX::RotationX "RotationX::RotationX(double angle)
";

%feature("docstring")  RotationX::clone "RotationX* RotationX::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  RotationX::createInverse "RotationX* RotationX::createInverse() const

Returns a new  IRotation object that is the current object's inverse. 
";

%feature("docstring")  RotationX::accept "void RotationX::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  RotationX::getAngle "double RotationX::getAngle() const 
";

%feature("docstring")  RotationX::getTransform3D "Transform3D RotationX::getTransform3D() const

Returns transformation. 
";


// File: classRotationY.xml
%feature("docstring") RotationY "";

%feature("docstring")  RotationY::RotationY "RotationY::RotationY(double angle)
";

%feature("docstring")  RotationY::clone "RotationY* RotationY::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  RotationY::createInverse "RotationY* RotationY::createInverse() const

Returns a new  IRotation object that is the current object's inverse. 
";

%feature("docstring")  RotationY::accept "void RotationY::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  RotationY::getAngle "double RotationY::getAngle() const 
";

%feature("docstring")  RotationY::getTransform3D "Transform3D RotationY::getTransform3D() const

Returns transformation. 
";


// File: classRotationZ.xml
%feature("docstring") RotationZ "";

%feature("docstring")  RotationZ::RotationZ "RotationZ::RotationZ(double angle=0.0)
";

%feature("docstring")  RotationZ::clone "RotationZ* RotationZ::clone() const

Returns a clone of this  ISample object. 
";

%feature("docstring")  RotationZ::createInverse "RotationZ* RotationZ::createInverse() const

Returns a new  IRotation object that is the current object's inverse. 
";

%feature("docstring")  RotationZ::accept "void RotationZ::accept(INodeVisitor *visitor) const

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  RotationZ::getAngle "double RotationZ::getAngle() const 
";

%feature("docstring")  RotationZ::getTransform3D "Transform3D RotationZ::getTransform3D() const

Returns transformation. 
";


// File: classRoughMultiLayerComputation.xml
%feature("docstring") RoughMultiLayerComputation "

Computes the diffuse reflection from the rough interfaces of a multilayer. Controlled by  MainComputation.

C++ includes: RoughMultiLayerComputation.h
";

%feature("docstring")  RoughMultiLayerComputation::RoughMultiLayerComputation "RoughMultiLayerComputation::RoughMultiLayerComputation(const MultiLayer *p_multi_layer)
";

%feature("docstring")  RoughMultiLayerComputation::~RoughMultiLayerComputation "RoughMultiLayerComputation::~RoughMultiLayerComputation()
";

%feature("docstring")  RoughMultiLayerComputation::eval "bool RoughMultiLayerComputation::eval(const SimulationOptions &options, ProgressHandler *progress, bool polarized, const std::vector< SimulationElement >::iterator &begin_it, const std::vector< SimulationElement >::iterator &end_it) const override

Calculate scattering intensity for each  SimulationElement returns false if nothing needed to be calculated 
";


// File: classExceptions_1_1RuntimeErrorException.xml
%feature("docstring") Exceptions::RuntimeErrorException "";

%feature("docstring")  Exceptions::RuntimeErrorException::RuntimeErrorException "Exceptions::RuntimeErrorException::RuntimeErrorException(const std::string &message)
";


// File: classSafePointerVector.xml
%feature("docstring") SafePointerVector "

The objects pointed to must support the  ICloneable interface.

A vector of pointers, owned by *this, with methods to handle them safely.

C++ includes: SafePointerVector.h
";

%feature("docstring")  SafePointerVector::SafePointerVector "SafePointerVector< T >::SafePointerVector()
";

%feature("docstring")  SafePointerVector::SafePointerVector "SafePointerVector< T >::SafePointerVector(const SafePointerVector &other)
";

%feature("docstring")  SafePointerVector::~SafePointerVector "virtual SafePointerVector< T >::~SafePointerVector()
";

%feature("docstring")  SafePointerVector::size "size_t SafePointerVector< T >::size() const 
";

%feature("docstring")  SafePointerVector::empty "bool SafePointerVector< T >::empty() const 
";

%feature("docstring")  SafePointerVector::push_back "void SafePointerVector< T >::push_back(T *pointer)
";

%feature("docstring")  SafePointerVector::begin "iterator SafePointerVector< T >::begin()
";

%feature("docstring")  SafePointerVector::begin "const_iterator SafePointerVector< T >::begin() const 
";

%feature("docstring")  SafePointerVector::end "iterator SafePointerVector< T >::end()
";

%feature("docstring")  SafePointerVector::end "const_iterator SafePointerVector< T >::end() const 
";

%feature("docstring")  SafePointerVector::deleteElement "bool SafePointerVector< T >::deleteElement(T *pointer)
";

%feature("docstring")  SafePointerVector::back "T* SafePointerVector< T >::back()
";

%feature("docstring")  SafePointerVector::back "const T* SafePointerVector< T >::back() const 
";

%feature("docstring")  SafePointerVector::clear "void SafePointerVector< T >::clear()
";


// File: classSampleBuilderFactory.xml
%feature("docstring") SampleBuilderFactory "

Factory to create standard pre-defined samples

C++ includes: SampleBuilderFactory.h
";

%feature("docstring")  SampleBuilderFactory::SampleBuilderFactory "SampleBuilderFactory::SampleBuilderFactory()
";

%feature("docstring")  SampleBuilderFactory::createSample "MultiLayer * SampleBuilderFactory::createSample(const std::string &name)

Retrieves a SampleBuilder from the registry, does the build, and returns the result. 
";


// File: classSampleIteratorPostorderStrategy.xml
%feature("docstring") SampleIteratorPostorderStrategy "";

%feature("docstring")  SampleIteratorPostorderStrategy::SampleIteratorPostorderStrategy "SampleIteratorPostorderStrategy::SampleIteratorPostorderStrategy()
";

%feature("docstring")  SampleIteratorPostorderStrategy::clone "SampleIteratorPostorderStrategy * SampleIteratorPostorderStrategy::clone() const 
";

%feature("docstring")  SampleIteratorPostorderStrategy::~SampleIteratorPostorderStrategy "SampleIteratorPostorderStrategy::~SampleIteratorPostorderStrategy()
";

%feature("docstring")  SampleIteratorPostorderStrategy::first "IteratorMemento SampleIteratorPostorderStrategy::first(const INode *p_root)
";

%feature("docstring")  SampleIteratorPostorderStrategy::next "void SampleIteratorPostorderStrategy::next(IteratorMemento &iterator_stack) const 
";

%feature("docstring")  SampleIteratorPostorderStrategy::isDone "bool SampleIteratorPostorderStrategy::isDone(IteratorMemento &iterator_stack) const 
";


// File: classSampleIteratorPreorderStrategy.xml
%feature("docstring") SampleIteratorPreorderStrategy "";

%feature("docstring")  SampleIteratorPreorderStrategy::SampleIteratorPreorderStrategy "SampleIteratorPreorderStrategy::SampleIteratorPreorderStrategy()
";

%feature("docstring")  SampleIteratorPreorderStrategy::clone "SampleIteratorPreorderStrategy * SampleIteratorPreorderStrategy::clone() const 
";

%feature("docstring")  SampleIteratorPreorderStrategy::~SampleIteratorPreorderStrategy "SampleIteratorPreorderStrategy::~SampleIteratorPreorderStrategy()
";

%feature("docstring")  SampleIteratorPreorderStrategy::first "IteratorMemento SampleIteratorPreorderStrategy::first(const INode *p_root)
";

%feature("docstring")  SampleIteratorPreorderStrategy::next "void SampleIteratorPreorderStrategy::next(IteratorMemento &iterator_stack) const 
";

%feature("docstring")  SampleIteratorPreorderStrategy::isDone "bool SampleIteratorPreorderStrategy::isDone(IteratorMemento &iterator_stack) const 
";


// File: classSampleLabelHandler.xml
%feature("docstring") SampleLabelHandler "

The handler which construct labels for sample variables during python script generation.

C++ includes: SampleLabelHandler.h
";

%feature("docstring")  SampleLabelHandler::SampleLabelHandler "SampleLabelHandler::SampleLabelHandler()
";

%feature("docstring")  SampleLabelHandler::getFormFactorMap "formfactors_t* SampleLabelHandler::getFormFactorMap()
";

%feature("docstring")  SampleLabelHandler::getInterferenceFunctionMap "interferences_t* SampleLabelHandler::getInterferenceFunctionMap()
";

%feature("docstring")  SampleLabelHandler::getLayerMap "layers_t* SampleLabelHandler::getLayerMap()
";

%feature("docstring")  SampleLabelHandler::getParticleLayoutMap "layouts_t* SampleLabelHandler::getParticleLayoutMap()
";

%feature("docstring")  SampleLabelHandler::getMaterialMap "materials_t* SampleLabelHandler::getMaterialMap()
";

%feature("docstring")  SampleLabelHandler::getMultiLayerMap "multilayers_t* SampleLabelHandler::getMultiLayerMap()
";

%feature("docstring")  SampleLabelHandler::getParticleCompositionMap "particlecompositions_t* SampleLabelHandler::getParticleCompositionMap()
";

%feature("docstring")  SampleLabelHandler::getParticleDistributionsMap "particledistributions_t* SampleLabelHandler::getParticleDistributionsMap()
";

%feature("docstring")  SampleLabelHandler::getParticleMap "particles_t* SampleLabelHandler::getParticleMap()
";

%feature("docstring")  SampleLabelHandler::getParticleCoreShellMap "particlescoreshell_t* SampleLabelHandler::getParticleCoreShellMap()
";

%feature("docstring")  SampleLabelHandler::getRotationsMap "rotations_t* SampleLabelHandler::getRotationsMap()
";

%feature("docstring")  SampleLabelHandler::getLayerRoughnessMap "roughnesses_t* SampleLabelHandler::getLayerRoughnessMap()
";

%feature("docstring")  SampleLabelHandler::getLabelParticle "std::string SampleLabelHandler::getLabelParticle(const IAbstractParticle *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelFormFactor "std::string SampleLabelHandler::getLabelFormFactor(const IFormFactor *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelInterferenceFunction "std::string SampleLabelHandler::getLabelInterferenceFunction(const IInterferenceFunction *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelLayout "std::string SampleLabelHandler::getLabelLayout(const ILayout *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelMaterial "std::string SampleLabelHandler::getLabelMaterial(const IMaterial *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelRotation "std::string SampleLabelHandler::getLabelRotation(const IRotation *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelLayer "std::string SampleLabelHandler::getLabelLayer(const Layer *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelRoughness "std::string SampleLabelHandler::getLabelRoughness(const LayerRoughness *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelMultiLayer "std::string SampleLabelHandler::getLabelMultiLayer(const MultiLayer *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelParticleComposition "std::string SampleLabelHandler::getLabelParticleComposition(const ParticleComposition *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelParticleCoreShell "std::string SampleLabelHandler::getLabelParticleCoreShell(const ParticleCoreShell *sample)
";

%feature("docstring")  SampleLabelHandler::getLabelParticleDistribution "std::string SampleLabelHandler::getLabelParticleDistribution(const ParticleDistribution *sample)
";

%feature("docstring")  SampleLabelHandler::insertMaterial "void SampleLabelHandler::insertMaterial(const IMaterial *sample)
";

%feature("docstring")  SampleLabelHandler::insertFormFactor "void SampleLabelHandler::insertFormFactor(const IFormFactor *sample)
";

%feature("docstring")  SampleLabelHandler::insertInterferenceFunction "void SampleLabelHandler::insertInterferenceFunction(const IInterferenceFunction *sample)
";

%feature("docstring")  SampleLabelHandler::insertLayout "void SampleLabelHandler::insertLayout(const ILayout *sample)
";

%feature("docstring")  SampleLabelHandler::insertRotation "void SampleLabelHandler::insertRotation(const IRotation *sample)
";

%feature("docstring")  SampleLabelHandler::insertLayer "void SampleLabelHandler::insertLayer(const Layer *sample)
";

%feature("docstring")  SampleLabelHandler::insertRoughness "void SampleLabelHandler::insertRoughness(const LayerRoughness *sample)
";

%feature("docstring")  SampleLabelHandler::insertMultiLayer "void SampleLabelHandler::insertMultiLayer(const MultiLayer *sample)
";

%feature("docstring")  SampleLabelHandler::insertParticle "void SampleLabelHandler::insertParticle(const Particle *sample)
";

%feature("docstring")  SampleLabelHandler::insertParticleComposition "void SampleLabelHandler::insertParticleComposition(const ParticleComposition *sample)
";

%feature("docstring")  SampleLabelHandler::insertParticleCoreShell "void SampleLabelHandler::insertParticleCoreShell(const ParticleCoreShell *sample)
";

%feature("docstring")  SampleLabelHandler::insertParticleDistribution "void SampleLabelHandler::insertParticleDistribution(const ParticleDistribution *sample)
";


// File: classSampleTreeIterator.xml
%feature("docstring") SampleTreeIterator "

Iterator through  INode tree of objects.

Usage example: SampleTreeIterator<Strategy> it(&sample); it.first(); while( !it.is_done() ) {  INode *p_sample = it.get_current(); it.next(); }

C++ includes: SampleTreeIterator.h
";

%feature("docstring")  SampleTreeIterator::SampleTreeIterator "SampleTreeIterator< Strategy >::SampleTreeIterator(const INode *root)
";

%feature("docstring")  SampleTreeIterator::~SampleTreeIterator "virtual SampleTreeIterator< Strategy >::~SampleTreeIterator()
";

%feature("docstring")  SampleTreeIterator::first "void SampleTreeIterator< Strategy >::first()
";

%feature("docstring")  SampleTreeIterator::next "void SampleTreeIterator< Strategy >::next()
";

%feature("docstring")  SampleTreeIterator::getCurrent "const INode * SampleTreeIterator< Strategy >::getCurrent()
";

%feature("docstring")  SampleTreeIterator::isDone "bool SampleTreeIterator< Strategy >::isDone() const 
";

%feature("docstring")  SampleTreeIterator::depth "size_t SampleTreeIterator< Strategy >::depth() const 
";


// File: classScalarRTCoefficients.xml
%feature("docstring") ScalarRTCoefficients "

Specular reflection and transmission coefficients in a layer in case of scalar interactions between the layers and the scattered particle.

C++ includes: ScalarRTCoefficients.h
";

%feature("docstring")  ScalarRTCoefficients::ScalarRTCoefficients "ScalarRTCoefficients::ScalarRTCoefficients()
";

%feature("docstring")  ScalarRTCoefficients::~ScalarRTCoefficients "virtual ScalarRTCoefficients::~ScalarRTCoefficients()
";

%feature("docstring")  ScalarRTCoefficients::clone "ScalarRTCoefficients * ScalarRTCoefficients::clone() const 
";

%feature("docstring")  ScalarRTCoefficients::T1plus "Eigen::Vector2cd ScalarRTCoefficients::T1plus() const

The following functions return the transmitted and reflected amplitudes for different incoming beam polarizations and eigenmodes 
";

%feature("docstring")  ScalarRTCoefficients::R1plus "Eigen::Vector2cd ScalarRTCoefficients::R1plus() const 
";

%feature("docstring")  ScalarRTCoefficients::T2plus "Eigen::Vector2cd ScalarRTCoefficients::T2plus() const 
";

%feature("docstring")  ScalarRTCoefficients::R2plus "Eigen::Vector2cd ScalarRTCoefficients::R2plus() const 
";

%feature("docstring")  ScalarRTCoefficients::T1min "Eigen::Vector2cd ScalarRTCoefficients::T1min() const 
";

%feature("docstring")  ScalarRTCoefficients::R1min "Eigen::Vector2cd ScalarRTCoefficients::R1min() const 
";

%feature("docstring")  ScalarRTCoefficients::T2min "Eigen::Vector2cd ScalarRTCoefficients::T2min() const 
";

%feature("docstring")  ScalarRTCoefficients::R2min "Eigen::Vector2cd ScalarRTCoefficients::R2min() const 
";

%feature("docstring")  ScalarRTCoefficients::getKz "Eigen::Vector2cd ScalarRTCoefficients::getKz() const

Returns z-part of the two wavevector eigenmodes. 
";

%feature("docstring")  ScalarRTCoefficients::getScalarT "complex_t ScalarRTCoefficients::getScalarT() const

Scalar value getters; these throw errors by default as they should only be used when the derived object is really scalar 
";

%feature("docstring")  ScalarRTCoefficients::getScalarR "complex_t ScalarRTCoefficients::getScalarR() const 
";

%feature("docstring")  ScalarRTCoefficients::getScalarKz "virtual complex_t ScalarRTCoefficients::getScalarKz() const 
";


// File: classScalarSpecularInfoMap.xml
%feature("docstring") ScalarSpecularInfoMap "

Implementation of ISpecularInfoMap for scalar valued reflection/transmission coefficients.

C++ includes: ScalarSpecularInfoMap.h
";

%feature("docstring")  ScalarSpecularInfoMap::ScalarSpecularInfoMap "ScalarSpecularInfoMap::ScalarSpecularInfoMap(const MultiLayer *multilayer, size_t layer_index)
";

%feature("docstring")  ScalarSpecularInfoMap::~ScalarSpecularInfoMap "ScalarSpecularInfoMap::~ScalarSpecularInfoMap() final
";

%feature("docstring")  ScalarSpecularInfoMap::clone "ScalarSpecularInfoMap* ScalarSpecularInfoMap::clone() const finaloverride
";

%feature("docstring")  ScalarSpecularInfoMap::getOutCoefficients "const ILayerRTCoefficients * ScalarSpecularInfoMap::getOutCoefficients(const SimulationElement &sim_element) const finaloverride

Retrieves the amplitude coefficients for the given angles. 
";

%feature("docstring")  ScalarSpecularInfoMap::getInCoefficients "const ILayerRTCoefficients * ScalarSpecularInfoMap::getInCoefficients(const SimulationElement &sim_element) const finaloverride

Retrieves the amplitude coefficients for the given angles. 
";


// File: classExceptions_1_1SelfReferenceException.xml
%feature("docstring") Exceptions::SelfReferenceException "";

%feature("docstring")  Exceptions::SelfReferenceException::SelfReferenceException "Exceptions::SelfReferenceException::SelfReferenceException(const std::string &message)
";


// File: classSimpleSelectionRule.xml
%feature("docstring") SimpleSelectionRule "

Selection rule (v*q)modulus!=0, defined by vector v(a,b,c) and modulus.

C++ includes: ISelectionRule.h
";

%feature("docstring")  SimpleSelectionRule::SimpleSelectionRule "SimpleSelectionRule::SimpleSelectionRule(int a, int b, int c, int modulus)
";

%feature("docstring")  SimpleSelectionRule::~SimpleSelectionRule "virtual SimpleSelectionRule::~SimpleSelectionRule()
";

%feature("docstring")  SimpleSelectionRule::clone "SimpleSelectionRule * SimpleSelectionRule::clone() const 
";

%feature("docstring")  SimpleSelectionRule::coordinateSelected "bool SimpleSelectionRule::coordinateSelected(const ivector_t &coordinate) const 
";


// File: classSimulation.xml
%feature("docstring") Simulation "

Pure virtual base class of OffSpecularSimulation and  GISASSimulation, holds common infrastructure to run a simulation.

C++ includes: Simulation.h
";

%feature("docstring")  Simulation::Simulation "Simulation::Simulation()
";

%feature("docstring")  Simulation::Simulation "Simulation::Simulation(const MultiLayer &p_sample)
";

%feature("docstring")  Simulation::Simulation "Simulation::Simulation(const std::shared_ptr< IMultiLayerBuilder > p_sample_builder)
";

%feature("docstring")  Simulation::~Simulation "Simulation::~Simulation()
";

%feature("docstring")  Simulation::clone "virtual Simulation* Simulation::clone() const =0
";

%feature("docstring")  Simulation::prepareSimulation "void Simulation::prepareSimulation()

Put into a clean state for running a simulation. 
";

%feature("docstring")  Simulation::runSimulation "void Simulation::runSimulation()

Run a simulation, possibly averaged over parameter distributions.

Run simulation with possible averaging over parameter distributions. 
";

%feature("docstring")  Simulation::setInstrument "void Simulation::setInstrument(const Instrument &instrument)
";

%feature("docstring")  Simulation::getInstrument "const Instrument& Simulation::getInstrument() const 
";

%feature("docstring")  Simulation::getInstrument "Instrument& Simulation::getInstrument()
";

%feature("docstring")  Simulation::setBeamIntensity "void Simulation::setBeamIntensity(double intensity)
";

%feature("docstring")  Simulation::getBeamIntensity "double Simulation::getBeamIntensity() const 
";

%feature("docstring")  Simulation::setBeamPolarization "void Simulation::setBeamPolarization(const kvector_t bloch_vector)

Sets the beam polarization according to the given Bloch vector. 
";

%feature("docstring")  Simulation::setDetectorResolutionFunction "void Simulation::setDetectorResolutionFunction(const IResolutionFunction2D &resolution_function)
";

%feature("docstring")  Simulation::removeDetectorResolutionFunction "void Simulation::removeDetectorResolutionFunction()
";

%feature("docstring")  Simulation::setAnalyzerProperties "void Simulation::setAnalyzerProperties(const kvector_t direction, double efficiency, double total_transmission)

Sets the polarization analyzer characteristics of the detector. 
";

%feature("docstring")  Simulation::setSample "void Simulation::setSample(const MultiLayer &sample)

The  MultiLayer object will not be owned by the  Simulation object. 
";

%feature("docstring")  Simulation::getSample "MultiLayer* Simulation::getSample() const 
";

%feature("docstring")  Simulation::setSampleBuilder "void Simulation::setSampleBuilder(const std::shared_ptr< IMultiLayerBuilder > sample_builder)
";

%feature("docstring")  Simulation::getSampleBuilder "std::shared_ptr<IMultiLayerBuilder> Simulation::getSampleBuilder() const 
";

%feature("docstring")  Simulation::numberOfSimulationElements "virtual int Simulation::numberOfSimulationElements() const =0
";

%feature("docstring")  Simulation::getDetectorIntensity "virtual OutputData<double>* Simulation::getDetectorIntensity(IDetector2D::EAxesUnits units_type=IDetector2D::DEFAULT) const =0

Clone simulated intensity map. 
";

%feature("docstring")  Simulation::addParameterDistribution "void Simulation::addParameterDistribution(const std::string &param_name, const IDistribution1D &distribution, size_t nbr_samples, double sigma_factor=0.0, const RealLimits &limits=RealLimits())
";

%feature("docstring")  Simulation::addParameterDistribution "void Simulation::addParameterDistribution(const ParameterDistribution &par_distr)
";

%feature("docstring")  Simulation::getDistributionHandler "const DistributionHandler& Simulation::getDistributionHandler() const 
";

%feature("docstring")  Simulation::setOptions "void Simulation::setOptions(const SimulationOptions &options)
";

%feature("docstring")  Simulation::getOptions "const SimulationOptions& Simulation::getOptions() const 
";

%feature("docstring")  Simulation::getOptions "SimulationOptions& Simulation::getOptions()
";

%feature("docstring")  Simulation::subscribe "void Simulation::subscribe(ProgressHandler::Callback_t inform)
";

%feature("docstring")  Simulation::setTerminalProgressMonitor "void Simulation::setTerminalProgressMonitor()

Initializes a progress monitor that prints to stdout. 
";

%feature("docstring")  Simulation::getChildren "std::vector< const INode * > Simulation::getChildren() const

Returns a vector of children (const). 
";


// File: classSimulationArea.xml
%feature("docstring") SimulationArea "

Holds iteration logic over active detector channels in the presence of masked areas and  RegionOfInterest defined.

C++ includes: SimulationArea.h
";

%feature("docstring")  SimulationArea::SimulationArea "SimulationArea::SimulationArea(const IDetector2D *detector)
";

%feature("docstring")  SimulationArea::~SimulationArea "virtual SimulationArea::~SimulationArea()
";

%feature("docstring")  SimulationArea::begin "SimulationAreaIterator SimulationArea::begin()
";

%feature("docstring")  SimulationArea::end "SimulationAreaIterator SimulationArea::end()
";

%feature("docstring")  SimulationArea::isMasked "bool SimulationArea::isMasked(size_t index) const

returns trus if given iterator index correspond to masked detector channel 
";

%feature("docstring")  SimulationArea::totalSize "size_t SimulationArea::totalSize() const 
";

%feature("docstring")  SimulationArea::roiIndex "size_t SimulationArea::roiIndex(size_t index) const

Return index in ROI map from iterator index. 
";

%feature("docstring")  SimulationArea::detectorIndex "size_t SimulationArea::detectorIndex(size_t index) const

Return detector index from iterator index. 
";


// File: classSimulationAreaIterator.xml
%feature("docstring") SimulationAreaIterator "

An iterator for  SimulationArea.

C++ includes: SimulationAreaIterator.h
";

%feature("docstring")  SimulationAreaIterator::SimulationAreaIterator "SimulationAreaIterator::SimulationAreaIterator(const SimulationArea *area, size_t start_at_index)
";

%feature("docstring")  SimulationAreaIterator::index "size_t SimulationAreaIterator::index() const 
";

%feature("docstring")  SimulationAreaIterator::elementIndex "size_t SimulationAreaIterator::elementIndex() const 
";

%feature("docstring")  SimulationAreaIterator::roiIndex "int SimulationAreaIterator::roiIndex() const 
";

%feature("docstring")  SimulationAreaIterator::detectorIndex "int SimulationAreaIterator::detectorIndex() const 
";


// File: classSimulationElement.xml
%feature("docstring") SimulationElement "

Data stucture containing both input and output of a single detector cell.

C++ includes: SimulationElement.h
";

%feature("docstring")  SimulationElement::SimulationElement "SimulationElement::SimulationElement(double wavelength, double alpha_i, double phi_i, std::unique_ptr< IPixel > pixel)
";

%feature("docstring")  SimulationElement::SimulationElement "SimulationElement::SimulationElement(const SimulationElement &other)
";

%feature("docstring")  SimulationElement::SimulationElement "SimulationElement::SimulationElement(const SimulationElement &other, double x, double y)

Construct  SimulationElement from other element and restrict k_f to specific value in the original detector pixel 
";

%feature("docstring")  SimulationElement::SimulationElement "SimulationElement::SimulationElement(SimulationElement &&other) noexcept
";

%feature("docstring")  SimulationElement::~SimulationElement "SimulationElement::~SimulationElement()
";

%feature("docstring")  SimulationElement::setPolarization "void SimulationElement::setPolarization(const Eigen::Matrix2cd &polarization)

Sets the polarization density matrix (in spin basis along z-axis) 
";

%feature("docstring")  SimulationElement::getPolarization "Eigen::Matrix2cd SimulationElement::getPolarization() const

Gets the polarization density matrix (in spin basis along z-axis) 
";

%feature("docstring")  SimulationElement::setAnalyzerOperator "void SimulationElement::setAnalyzerOperator(const Eigen::Matrix2cd &polarization_operator)

Sets the polarization analyzer operator (in spin basis along z-axis) 
";

%feature("docstring")  SimulationElement::getAnalyzerOperator "Eigen::Matrix2cd SimulationElement::getAnalyzerOperator() const

Gets the polarization analyzer operator (in spin basis along z-axis) 
";

%feature("docstring")  SimulationElement::getWavelength "double SimulationElement::getWavelength() const 
";

%feature("docstring")  SimulationElement::getAlphaI "double SimulationElement::getAlphaI() const 
";

%feature("docstring")  SimulationElement::getPhiI "double SimulationElement::getPhiI() const 
";

%feature("docstring")  SimulationElement::getAlphaMean "double SimulationElement::getAlphaMean() const 
";

%feature("docstring")  SimulationElement::getPhiMean "double SimulationElement::getPhiMean() const 
";

%feature("docstring")  SimulationElement::setIntensity "void SimulationElement::setIntensity(double intensity)
";

%feature("docstring")  SimulationElement::addIntensity "void SimulationElement::addIntensity(double intensity)
";

%feature("docstring")  SimulationElement::getIntensity "double SimulationElement::getIntensity() const 
";

%feature("docstring")  SimulationElement::getKi "kvector_t SimulationElement::getKi() const 
";

%feature("docstring")  SimulationElement::getMeanKf "kvector_t SimulationElement::getMeanKf() const 
";

%feature("docstring")  SimulationElement::getMeanQ "kvector_t SimulationElement::getMeanQ() const 
";

%feature("docstring")  SimulationElement::getQ "kvector_t SimulationElement::getQ(double x, double y) const

Returns scattering vector Q, with Kf determined from in-pixel coordinates x,y. In-pixel coordinates take values from 0 to 1. 
";

%feature("docstring")  SimulationElement::getIntegrationFactor "double SimulationElement::getIntegrationFactor(double x, double y) const 
";

%feature("docstring")  SimulationElement::getSolidAngle "double SimulationElement::getSolidAngle() const 
";

%feature("docstring")  SimulationElement::getAlpha "double SimulationElement::getAlpha(double x, double y) const 
";

%feature("docstring")  SimulationElement::getPhi "double SimulationElement::getPhi(double x, double y) const 
";

%feature("docstring")  SimulationElement::containsSpecularWavevector "bool SimulationElement::containsSpecularWavevector() const

check if element contains given wavevector 
";

%feature("docstring")  SimulationElement::setSpecular "void SimulationElement::setSpecular(bool contains_specular)

indicate that this element contains the specular wavevector 
";


// File: classSimulationFactory.xml
%feature("docstring") SimulationFactory "

Registry to create standard pre-defined simulations. Used in functional tests, performance measurements, etc.

C++ includes: SimulationFactory.h
";

%feature("docstring")  SimulationFactory::SimulationFactory "SimulationFactory::SimulationFactory()
";


// File: classSimulationOptions.xml
%feature("docstring") SimulationOptions "

Collect the different options for simulation.

SimulationOptions

C++ includes: SimulationOptions.h
";

%feature("docstring")  SimulationOptions::SimulationOptions "SimulationOptions::SimulationOptions()
";

%feature("docstring")  SimulationOptions::isIntegrate "bool SimulationOptions::isIntegrate() const 
";

%feature("docstring")  SimulationOptions::getMcPoints "size_t SimulationOptions::getMcPoints() const 
";

%feature("docstring")  SimulationOptions::setMonteCarloIntegration "void SimulationOptions::setMonteCarloIntegration(bool flag=true, size_t mc_points=50)

Enables/disables MonetCarlo integration.

Parameters:
-----------

flag: 
If true, MonteCarlo integration will be used, otherwise analytical calculations

mc_points: 
Number of points for MonteCarlo integrator 
";

%feature("docstring")  SimulationOptions::setNumberOfThreads "void SimulationOptions::setNumberOfThreads(int nthreads)

Sets number of threads to use during the simulation (0 - take the default value from the hardware) 
";

%feature("docstring")  SimulationOptions::getNumberOfThreads "int SimulationOptions::getNumberOfThreads() const 
";

%feature("docstring")  SimulationOptions::setNumberOfBatches "void SimulationOptions::setNumberOfBatches(int nbatches)

Sets number of batches to split. 
";

%feature("docstring")  SimulationOptions::getNumberOfBatches "int SimulationOptions::getNumberOfBatches() const 
";

%feature("docstring")  SimulationOptions::getCurrentBatch "int SimulationOptions::getCurrentBatch() const 
";

%feature("docstring")  SimulationOptions::setThreadInfo "void SimulationOptions::setThreadInfo(const ThreadInfo &thread_info)

Sets the batch and thread information to be used. 
";

%feature("docstring")  SimulationOptions::getHardwareConcurrency "int SimulationOptions::getHardwareConcurrency() const 
";

%feature("docstring")  SimulationOptions::setIncludeSpecular "void SimulationOptions::setIncludeSpecular(bool include_specular)
";

%feature("docstring")  SimulationOptions::includeSpecular "bool SimulationOptions::includeSpecular() const 
";


// File: classSimulationRoiArea.xml
%feature("docstring") SimulationRoiArea "

Holds iteration logic over active detector channels in the presence of ROI. On the contrary to  SimulationArea class, iterates also over masked areas.

C++ includes: SimulationArea.h
";

%feature("docstring")  SimulationRoiArea::SimulationRoiArea "SimulationRoiArea::SimulationRoiArea(const IDetector2D *detector)
";

%feature("docstring")  SimulationRoiArea::isMasked "bool SimulationRoiArea::isMasked(size_t) const

returns trus if given iterator index correspond to masked detector channel 
";


// File: classSizeDistributionDAModelBuilder.xml
%feature("docstring") SizeDistributionDAModelBuilder "

Creates the sample demonstrating size distribution model in decoupling approximation. Equivalent of Examples/python/simulation/ex03_InterferenceFunctions/ApproximationDA.py

C++ includes: SizeDistributionModelsBuilder.h
";

%feature("docstring")  SizeDistributionDAModelBuilder::SizeDistributionDAModelBuilder "SizeDistributionDAModelBuilder::SizeDistributionDAModelBuilder()
";

%feature("docstring")  SizeDistributionDAModelBuilder::buildSample "MultiLayer * SizeDistributionDAModelBuilder::buildSample() const 
";


// File: classSizeDistributionLMAModelBuilder.xml
%feature("docstring") SizeDistributionLMAModelBuilder "

Creates the sample demonstrating size distribution model in local monodisperse approximation. Equivalent of Examples/python/simulation/ex03_InterferenceFunctions/ApproximationLMA.py

C++ includes: SizeDistributionModelsBuilder.h
";

%feature("docstring")  SizeDistributionLMAModelBuilder::SizeDistributionLMAModelBuilder "SizeDistributionLMAModelBuilder::SizeDistributionLMAModelBuilder()
";

%feature("docstring")  SizeDistributionLMAModelBuilder::buildSample "MultiLayer * SizeDistributionLMAModelBuilder::buildSample() const 
";


// File: classSizeDistributionSSCAModelBuilder.xml
%feature("docstring") SizeDistributionSSCAModelBuilder "

Creates the sample demonstrating size distribution model in size space coupling approximation. Equivalent of Examples/python/simulation/ex03_InterferenceFunctions/ApproximationSSCA.py

C++ includes: SizeDistributionModelsBuilder.h
";

%feature("docstring")  SizeDistributionSSCAModelBuilder::SizeDistributionSSCAModelBuilder "SizeDistributionSSCAModelBuilder::SizeDistributionSSCAModelBuilder()
";

%feature("docstring")  SizeDistributionSSCAModelBuilder::buildSample "MultiLayer * SizeDistributionSSCAModelBuilder::buildSample() const 
";


// File: classSlice.xml
%feature("docstring") Slice "

Represents a single slice, constructed from the division of a layer into (possibly) multiple slices.

C++ includes: Slice.h
";

%feature("docstring")  Slice::Slice "Slice::Slice(const IMaterial &material, double thickness, const LayerRoughness *top_roughness=nullptr)
";

%feature("docstring")  Slice::~Slice "Slice::~Slice()
";


// File: classSpecularComputation.xml
%feature("docstring") SpecularComputation "

Computes the specular scattering. Controlled by  MainComputation.

C++ includes: SpecularComputation.h
";

%feature("docstring")  SpecularComputation::SpecularComputation "SpecularComputation::SpecularComputation(const MultiLayer *p_multi_layer)
";

%feature("docstring")  SpecularComputation::eval "bool SpecularComputation::eval(const SimulationOptions &options, ProgressHandler *progress, bool polarized, const std::vector< SimulationElement >::iterator &begin_it, const std::vector< SimulationElement >::iterator &end_it) const override

Calculate scattering intensity for each  SimulationElement returns false if nothing needed to be calculated 
";


// File: classSpecularMagnetic.xml
%feature("docstring") SpecularMagnetic "

Implements the matrix formalism for the calculation of wave amplitudes of the coherent wave solution in a multilayer with magnetization.

C++ includes: SpecularMagnetic.h
";


// File: classSpecularMatrix.xml
%feature("docstring") SpecularMatrix "

Implements method 'execute' to compute refraction angles and transmission/reflection coefficients for coherent wave propagation in a multilayer.

C++ includes: SpecularMatrix.h
";


// File: classSpecularSimulation.xml
%feature("docstring") SpecularSimulation "

Main class to run a specular simulation.

C++ includes: SpecularSimulation.h
";

%feature("docstring")  SpecularSimulation::SpecularSimulation "SpecularSimulation::SpecularSimulation()
";

%feature("docstring")  SpecularSimulation::SpecularSimulation "SpecularSimulation::SpecularSimulation(const ISample &sample)
";

%feature("docstring")  SpecularSimulation::SpecularSimulation "SpecularSimulation::SpecularSimulation(const std::shared_ptr< IMultiLayerBuilder > sample_builder)
";

%feature("docstring")  SpecularSimulation::~SpecularSimulation "SpecularSimulation::~SpecularSimulation()
";

%feature("docstring")  SpecularSimulation::clone "SpecularSimulation * SpecularSimulation::clone() const 
";

%feature("docstring")  SpecularSimulation::runSimulation "void SpecularSimulation::runSimulation()

Run a simulation with the current parameter settings. 
";

%feature("docstring")  SpecularSimulation::setSample "void SpecularSimulation::setSample(const ISample &sample)

Sets the sample to be tested. 
";

%feature("docstring")  SpecularSimulation::getSample "ISample* SpecularSimulation::getSample() const

Returns the sample. 
";

%feature("docstring")  SpecularSimulation::setSampleBuilder "void SpecularSimulation::setSampleBuilder(std::shared_ptr< IMultiLayerBuilder > sample_builder)

Sets the sample builder. 
";

%feature("docstring")  SpecularSimulation::getSampleBuilder "std::shared_ptr<IMultiLayerBuilder> SpecularSimulation::getSampleBuilder() const

return sample builder 
";

%feature("docstring")  SpecularSimulation::setBeamParameters "void SpecularSimulation::setBeamParameters(double lambda, const IAxis &alpha_axis)

Sets beam parameters with alpha_i of the beam defined in the range. 
";

%feature("docstring")  SpecularSimulation::setBeamParameters "void SpecularSimulation::setBeamParameters(double lambda, int nbins, double alpha_i_min, double alpha_i_max)
";

%feature("docstring")  SpecularSimulation::setEvanescentWaveAxis "void SpecularSimulation::setEvanescentWaveAxis(const IAxis &z_axis)

set axis for evanescent wave axis 
";

%feature("docstring")  SpecularSimulation::setEvanescentWaveAxis "void SpecularSimulation::setEvanescentWaveAxis(int nbins, double z_min, double z_max)
";

%feature("docstring")  SpecularSimulation::getAlphaAxis "const IAxis * SpecularSimulation::getAlphaAxis() const

returns alpha_i axis 
";

%feature("docstring")  SpecularSimulation::getScalarR "std::vector< complex_t > SpecularSimulation::getScalarR(size_t i_layer) const

returns vector of reflection coefficients for all alpha_i angles for given layer index 
";

%feature("docstring")  SpecularSimulation::getScalarT "std::vector< complex_t > SpecularSimulation::getScalarT(size_t i_layer) const

returns vector of transmission coefficients for all alpha_i angles for given layer index 
";

%feature("docstring")  SpecularSimulation::getScalarKz "std::vector< complex_t > SpecularSimulation::getScalarKz(size_t i_layer) const

returns vector of Kz coefficients for all alpha_i angles for given layer index 
";

%feature("docstring")  SpecularSimulation::getLayerRTCoefficients "SpecularSimulation::LayerRTCoefficients_t SpecularSimulation::getLayerRTCoefficients(size_t i_alpha, size_t i_layer) const 
";

%feature("docstring")  SpecularSimulation::prepareSimulation "void SpecularSimulation::prepareSimulation()

Put into a clean state for running a simulation. 
";


// File: classSphericalDetector.xml
%feature("docstring") SphericalDetector "

A spherical detector with axes and resolution function.  SphericalDetector

C++ includes: SphericalDetector.h
";

%feature("docstring")  SphericalDetector::SphericalDetector "SphericalDetector::SphericalDetector()
";

%feature("docstring")  SphericalDetector::SphericalDetector "SphericalDetector::SphericalDetector(size_t n_phi, double phi_min, double phi_max, size_t n_alpha, double alpha_min, double alpha_max)

Spherical detector constructor using angle ranges

Parameters:
-----------

n_phi: 
number of phi-axis bins

phi_min: 
low edge of first phi-bin

phi_max: 
upper edge of last phi-bin

n_alpha: 
number of alpha-axis bins

alpha_min: 
low edge of first alpha-bin

alpha_max: 
upper edge of last alpha-bin 
";

%feature("docstring")  SphericalDetector::SphericalDetector "SphericalDetector::SphericalDetector(const SphericalDetector &other)
";

%feature("docstring")  SphericalDetector::clone "SphericalDetector * SphericalDetector::clone() const override
";

%feature("docstring")  SphericalDetector::accept "void SphericalDetector::accept(INodeVisitor *visitor) const override

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  SphericalDetector::~SphericalDetector "SphericalDetector::~SphericalDetector() override
";

%feature("docstring")  SphericalDetector::getValidAxesUnits "std::vector< IDetector2D::EAxesUnits > SphericalDetector::getValidAxesUnits() const override

returns vector of valid axes units 
";

%feature("docstring")  SphericalDetector::getDefaultAxesUnits "IDetector2D::EAxesUnits SphericalDetector::getDefaultAxesUnits() const override

return default axes units 
";


// File: classSphericalPixel.xml
%feature("docstring") SphericalPixel "";

%feature("docstring")  SphericalPixel::SphericalPixel "SphericalPixel::SphericalPixel(Bin1D alpha_bin, Bin1D phi_bin)
";

%feature("docstring")  SphericalPixel::~SphericalPixel "virtual SphericalPixel::~SphericalPixel()
";

%feature("docstring")  SphericalPixel::clone "SphericalPixel * SphericalPixel::clone() const override
";

%feature("docstring")  SphericalPixel::createZeroSizePixel "SphericalPixel * SphericalPixel::createZeroSizePixel(double x, double y) const override
";

%feature("docstring")  SphericalPixel::getK "kvector_t SphericalPixel::getK(double x, double y, double wavelength) const override
";

%feature("docstring")  SphericalPixel::getIntegrationFactor "double SphericalPixel::getIntegrationFactor(double x, double y) const override
";

%feature("docstring")  SphericalPixel::getSolidAngle "double SphericalPixel::getSolidAngle() const override
";


// File: classSquaredFunctionDefault.xml
%feature("docstring") SquaredFunctionDefault "

Squared difference between two values. value = (a-b)*(a-b)/norm, where norm = max(b, 1.0), a = simulated values, b = real_values.

C++ includes: ISquaredFunction.h
";

%feature("docstring")  SquaredFunctionDefault::SquaredFunctionDefault "SquaredFunctionDefault::SquaredFunctionDefault()
";

%feature("docstring")  SquaredFunctionDefault::~SquaredFunctionDefault "virtual SquaredFunctionDefault::~SquaredFunctionDefault()
";

%feature("docstring")  SquaredFunctionDefault::clone "virtual SquaredFunctionDefault* SquaredFunctionDefault::clone() const 
";

%feature("docstring")  SquaredFunctionDefault::calculateSquaredDifference "virtual double SquaredFunctionDefault::calculateSquaredDifference(double real_value, double simulated_value) const 
";

%feature("docstring")  SquaredFunctionDefault::calculateSquaredError "virtual double SquaredFunctionDefault::calculateSquaredError(double real_value, double simulated_value=0) const 
";


// File: classSquaredFunctionGaussianError.xml
%feature("docstring") SquaredFunctionGaussianError "

Squared difference between two values with gaussian error. value = (a-b)*(a-b)/norm, where norm = sigma*sigma; sigma is set by user.

C++ includes: ISquaredFunction.h
";

%feature("docstring")  SquaredFunctionGaussianError::SquaredFunctionGaussianError "SquaredFunctionGaussianError::SquaredFunctionGaussianError(double sigma=0.01)
";

%feature("docstring")  SquaredFunctionGaussianError::~SquaredFunctionGaussianError "virtual SquaredFunctionGaussianError::~SquaredFunctionGaussianError()
";

%feature("docstring")  SquaredFunctionGaussianError::clone "virtual SquaredFunctionGaussianError* SquaredFunctionGaussianError::clone() const 
";

%feature("docstring")  SquaredFunctionGaussianError::calculateSquaredDifference "virtual double SquaredFunctionGaussianError::calculateSquaredDifference(double real_value, double simulated_value) const 
";

%feature("docstring")  SquaredFunctionGaussianError::calculateSquaredError "virtual double SquaredFunctionGaussianError::calculateSquaredError(double, double) const 
";


// File: classSquaredFunctionMeanSquaredError.xml
%feature("docstring") SquaredFunctionMeanSquaredError "

Squared difference between two values normalized by mean squared error. value = (a-b)*(a-b)/norm, where norm = sqrt(sigma1*sigma1 + sigma2*sigma2), sigma1=max(a, 1.0), sigma2=max(b,1.0)

C++ includes: ISquaredFunction.h
";

%feature("docstring")  SquaredFunctionMeanSquaredError::SquaredFunctionMeanSquaredError "SquaredFunctionMeanSquaredError::SquaredFunctionMeanSquaredError()
";

%feature("docstring")  SquaredFunctionMeanSquaredError::~SquaredFunctionMeanSquaredError "virtual SquaredFunctionMeanSquaredError::~SquaredFunctionMeanSquaredError()
";

%feature("docstring")  SquaredFunctionMeanSquaredError::clone "virtual SquaredFunctionMeanSquaredError* SquaredFunctionMeanSquaredError::clone() const 
";

%feature("docstring")  SquaredFunctionMeanSquaredError::calculateSquaredDifference "virtual double SquaredFunctionMeanSquaredError::calculateSquaredDifference(double real_value, double simulated_value) const 
";

%feature("docstring")  SquaredFunctionMeanSquaredError::calculateSquaredError "virtual double SquaredFunctionMeanSquaredError::calculateSquaredError(double real_value, double simulated_value) const 
";


// File: classSquaredFunctionSimError.xml
%feature("docstring") SquaredFunctionSimError "

Squared difference between two values. value = (a-b)*(a-b)/norm, where norm = max(a, 1.0), a = simulated values, b = real_values.

C++ includes: ISquaredFunction.h
";

%feature("docstring")  SquaredFunctionSimError::SquaredFunctionSimError "SquaredFunctionSimError::SquaredFunctionSimError()
";

%feature("docstring")  SquaredFunctionSimError::~SquaredFunctionSimError "virtual SquaredFunctionSimError::~SquaredFunctionSimError()
";

%feature("docstring")  SquaredFunctionSimError::clone "virtual SquaredFunctionSimError* SquaredFunctionSimError::clone() const 
";

%feature("docstring")  SquaredFunctionSimError::calculateSquaredDifference "virtual double SquaredFunctionSimError::calculateSquaredDifference(double real_value, double simulated_value) const 
";

%feature("docstring")  SquaredFunctionSimError::calculateSquaredError "virtual double SquaredFunctionSimError::calculateSquaredError(double, double simulated_value) const 
";


// File: classSquaredFunctionSystematicError.xml
%feature("docstring") SquaredFunctionSystematicError "

Squared difference between two values normalized by systematic error. value = (a-b)*(a-b)/norm, where norm = max(error, 1.0), error = b + (epsilon*b)**2.

C++ includes: ISquaredFunction.h
";

%feature("docstring")  SquaredFunctionSystematicError::SquaredFunctionSystematicError "SquaredFunctionSystematicError::SquaredFunctionSystematicError(double epsilon=0.08)
";

%feature("docstring")  SquaredFunctionSystematicError::~SquaredFunctionSystematicError "virtual SquaredFunctionSystematicError::~SquaredFunctionSystematicError()
";

%feature("docstring")  SquaredFunctionSystematicError::clone "virtual SquaredFunctionSystematicError* SquaredFunctionSystematicError::clone() const 
";

%feature("docstring")  SquaredFunctionSystematicError::calculateSquaredDifference "virtual double SquaredFunctionSystematicError::calculateSquaredDifference(double real_value, double simulated_value) const 
";

%feature("docstring")  SquaredFunctionSystematicError::calculateSquaredError "virtual double SquaredFunctionSystematicError::calculateSquaredError(double real_value, double) const 
";


// File: classSquareLattice.xml
%feature("docstring") SquareLattice "";

%feature("docstring")  SquareLattice::SquareLattice "SquareLattice::SquareLattice(double length, double rotation_angle=0.0)
";

%feature("docstring")  SquareLattice::clone "SquareLattice * SquareLattice::clone() const 
";

%feature("docstring")  SquareLattice::accept "void SquareLattice::accept(INodeVisitor *visitor) const final

Calls the  INodeVisitor's visit method. 
";

%feature("docstring")  SquareLattice::length2 "double SquareLattice::length2() const 
";


// File: classSquareLatticeBuilder.xml
%feature("docstring") SquareLatticeBuilder "

Builds sample: 2D lattice with different disorder (IsGISAXS example #6).

C++ includes: TwoDimLatticeBuilder.h
";

%feature("docstring")  SquareLatticeBuilder::SquareLatticeBuilder "SquareLatticeBuilder::SquareLatticeBuilder()
";

%feature("docstring")  SquareLatticeBuilder::buildSample "MultiLayer * SquareLatticeBuilder::buildSample() const 
";


// File: classSSCApproximationStrategy.xml
%feature("docstring") SSCApproximationStrategy "

Virtual base class for  SSCApproximationStrategy1 and  SSCApproximationStrategy2, which compute the total scalar/polarized scattering from a decorated layer in size-spacing correlation approximation.

C++ includes: SSCApproximationStrategy.h
";

%feature("docstring")  SSCApproximationStrategy::SSCApproximationStrategy "SSCApproximationStrategy::SSCApproximationStrategy(double kappa)
";


// File: classSSCApproximationStrategy1.xml
%feature("docstring") SSCApproximationStrategy1 "

Strategy class to compute the total scalar scattering from a decorated layer in size-spacing correlation approximation.

C++ includes: SSCApproximationStrategy.h
";

%feature("docstring")  SSCApproximationStrategy1::SSCApproximationStrategy1 "SSCApproximationStrategy1::SSCApproximationStrategy1(SimulationOptions sim_params, double kappa)
";


// File: classSSCApproximationStrategy2.xml
%feature("docstring") SSCApproximationStrategy2 "

Strategy class to compute the total polarized scattering from a decorated layer in size-spacing correlation approximation.

C++ includes: SSCApproximationStrategy.h
";

%feature("docstring")  SSCApproximationStrategy2::SSCApproximationStrategy2 "SSCApproximationStrategy2::SSCApproximationStrategy2(SimulationOptions sim_params, double kappa)
";


// File: structThreadInfo.xml
%feature("docstring") ThreadInfo "

Information to run simulation with dedicated number of threads.

C++ includes: ThreadInfo.h
";

%feature("docstring")  ThreadInfo::ThreadInfo "ThreadInfo::ThreadInfo()
";


// File: classTRange.xml
%feature("docstring") TRange "

An interval [lowerBound..upperBound[.

C++ includes: TRange.h
";

%feature("docstring")  TRange::TRange "TRange< T >::TRange(T lowerBound, T upperBound)
";

%feature("docstring")  TRange::~TRange "virtual TRange< T >::~TRange()
";

%feature("docstring")  TRange::getLowerBound "T TRange< T >::getLowerBound() const 
";

%feature("docstring")  TRange::getUpperBound "T TRange< T >::getUpperBound() const 
";

%feature("docstring")  TRange::getDifference "T TRange< T >::getDifference() const 
";

%feature("docstring")  TRange::inRange "bool TRange< T >::inRange(T value) const 
";


// File: classTransform3D.xml
%feature("docstring") Transform3D "

Vector transformations in three dimensions.

C++ includes: Transform3D.h
";

%feature("docstring")  Transform3D::Transform3D "Transform3D::Transform3D()

Constructs unit transformation. 
";

%feature("docstring")  Transform3D::Transform3D "Transform3D::Transform3D(const Transform3D &other)

Copy constructor. 
";

%feature("docstring")  Transform3D::~Transform3D "Transform3D::~Transform3D()

Destructor. 
";

%feature("docstring")  Transform3D::clone "Transform3D * Transform3D::clone() const

Clones the transformation. 
";

%feature("docstring")  Transform3D::calculateEulerAngles "void Transform3D::calculateEulerAngles(double *p_alpha, double *p_beta, double *p_gamma) const

Calculates the Euler angles corresponding to the rotation. 
";

%feature("docstring")  Transform3D::getInverse "Transform3D Transform3D::getInverse() const

Returns the inverse transformation. 
";

%feature("docstring")  Transform3D::transformed "BasicVector3D< double > Transform3D::transformed(const BasicVector3D< double > &v) const

Return transformed vector  v. 
";

%feature("docstring")  Transform3D::transformed "BasicVector3D< complex_t > Transform3D::transformed(const BasicVector3D< complex_t > &v) const

Return transformed vector  v. 
";

%feature("docstring")  Transform3D::transformedInverse "BasicVector3D< double > Transform3D::transformedInverse(const BasicVector3D< double > &v) const

Return transformed vector  v. 
";

%feature("docstring")  Transform3D::transformedInverse "BasicVector3D< complex_t > Transform3D::transformedInverse(const BasicVector3D< complex_t > &v) const

Return transformed vector  v. 
";

%feature("docstring")  Transform3D::getRotationType "Transform3D::ERotationType Transform3D::getRotationType() const

Retrieve the rotation type (general, around x, y or z-axis) 
";

%feature("docstring")  Transform3D::isIdentity "bool Transform3D::isIdentity() const

Determine if the transformation is trivial (identity) 
";

%feature("docstring")  Transform3D::print "void Transform3D::print(std::ostream &ostr) const 
";


// File: classTransformBoxBuilder.xml
%feature("docstring") TransformBoxBuilder "

Rotated box in 3 layers system.

C++ includes: TransformationsBuilder.h
";

%feature("docstring")  TransformBoxBuilder::TransformBoxBuilder "TransformBoxBuilder::TransformBoxBuilder()
";

%feature("docstring")  TransformBoxBuilder::buildSample "MultiLayer * TransformBoxBuilder::buildSample() const 
";


// File: classTriangularRippleBuilder.xml
%feature("docstring") TriangularRippleBuilder "

Builds sample: triangular ripple within the 1D-paracrystal model (from PRB 85, 235415, 2012).

C++ includes: RipplesBuilder.h
";

%feature("docstring")  TriangularRippleBuilder::TriangularRippleBuilder "TriangularRippleBuilder::TriangularRippleBuilder()
";

%feature("docstring")  TriangularRippleBuilder::buildSample "MultiLayer * TriangularRippleBuilder::buildSample() const 
";


// File: classTSampledRange.xml
%feature("docstring") TSampledRange "

An interval [lowerBound..upperBound[, and a number of samples.

C++ includes: TRange.h
";

%feature("docstring")  TSampledRange::TSampledRange "TSampledRange< T >::TSampledRange(size_t n_samples, T lowerBound, T upperBound)
";

%feature("docstring")  TSampledRange::getNSamples "size_t TSampledRange< T >::getNSamples() const 
";


// File: classTwoTypesCylindersDistributionBuilder.xml
%feature("docstring") TwoTypesCylindersDistributionBuilder "

Builds mixture of cylinder particles with different size distribution (IsGISAXS example #2)

C++ includes: ParticleDistributionsBuilder.h
";

%feature("docstring")  TwoTypesCylindersDistributionBuilder::TwoTypesCylindersDistributionBuilder "TwoTypesCylindersDistributionBuilder::TwoTypesCylindersDistributionBuilder()
";

%feature("docstring")  TwoTypesCylindersDistributionBuilder::buildSample "MultiLayer * TwoTypesCylindersDistributionBuilder::buildSample() const 
";


// File: classUnit.xml
%feature("docstring") Unit "

A physical unit.

C++ includes: Unit.h
";

%feature("docstring")  Unit::Unit "Unit::Unit(const std::string &name=\"\")
";

%feature("docstring")  Unit::setUnit "void Unit::setUnit(const std::string &name)
";


// File: classExceptions_1_1UnknownClassRegistrationException.xml
%feature("docstring") Exceptions::UnknownClassRegistrationException "";

%feature("docstring")  Exceptions::UnknownClassRegistrationException::UnknownClassRegistrationException "Exceptions::UnknownClassRegistrationException::UnknownClassRegistrationException(const std::string &message)
";


// File: classVariableBinAxis.xml
%feature("docstring") VariableBinAxis "

Axis with variable bin size.

C++ includes: VariableBinAxis.h
";

%feature("docstring")  VariableBinAxis::VariableBinAxis "VariableBinAxis::VariableBinAxis(const std::string &name, size_t nbins, const std::vector< double > &bin_boundaries)

VariableBinAxis constructor.

Parameters:
-----------

name: 
Axis name

nbins: 
number of bins

bin_boundaries: 
Array of size nbins+1 containing low-edges for each bin and upper edge of last bin. 
";

%feature("docstring")  VariableBinAxis::~VariableBinAxis "virtual VariableBinAxis::~VariableBinAxis()
";

%feature("docstring")  VariableBinAxis::clone "VariableBinAxis * VariableBinAxis::clone() const

clone function 
";

%feature("docstring")  VariableBinAxis::size "size_t VariableBinAxis::size() const

retrieve the number of bins 
";

%feature("docstring")  VariableBinAxis::getBin "Bin1D VariableBinAxis::getBin(size_t index) const

retrieve a 1d bin for the given index 
";

%feature("docstring")  VariableBinAxis::getMin "double VariableBinAxis::getMin() const

Returns value of first point of axis. 
";

%feature("docstring")  VariableBinAxis::getMax "double VariableBinAxis::getMax() const

Returns value of last point of axis. 
";

%feature("docstring")  VariableBinAxis::getBinCenter "double VariableBinAxis::getBinCenter(size_t index) const 
";

%feature("docstring")  VariableBinAxis::findClosestIndex "size_t VariableBinAxis::findClosestIndex(double value) const

find bin index which is best match for given value 
";

%feature("docstring")  VariableBinAxis::getBinCenters "std::vector< double > VariableBinAxis::getBinCenters() const 
";

%feature("docstring")  VariableBinAxis::getBinBoundaries "std::vector<double> VariableBinAxis::getBinBoundaries() const 
";

%feature("docstring")  VariableBinAxis::createClippedAxis "VariableBinAxis * VariableBinAxis::createClippedAxis(double left, double right) const

Creates a new clipped axis. 
";


// File: classVerticalLine.xml
%feature("docstring") VerticalLine "

An infinite vertical line.

C++ includes: Line.h
";

%feature("docstring")  VerticalLine::VerticalLine "VerticalLine::VerticalLine(double x)

Parameters:
-----------

x: 
The value at which it crosses x-axes 
";

%feature("docstring")  VerticalLine::clone "VerticalLine* VerticalLine::clone() const 
";

%feature("docstring")  VerticalLine::contains "bool VerticalLine::contains(double x, double y) const

Returns true if point with given coordinates is inside or on border of the shape. 
";

%feature("docstring")  VerticalLine::contains "bool VerticalLine::contains(const Bin1D &binx, const Bin1D &biny) const

Returns true if area defined by two bins is inside or on border of polygon (more precisely, if mid point of two bins satisfy this condition). 
";

%feature("docstring")  VerticalLine::getXpos "double VerticalLine::getXpos() const 
";


// File: classWavevectorInfo.xml
%feature("docstring") WavevectorInfo "

Holds all wavevector information relevant for calculating form factors.

C++ includes: WavevectorInfo.h
";

%feature("docstring")  WavevectorInfo::WavevectorInfo "WavevectorInfo::WavevectorInfo()
";

%feature("docstring")  WavevectorInfo::WavevectorInfo "WavevectorInfo::WavevectorInfo(cvector_t ki, cvector_t kf, double wavelength)
";

%feature("docstring")  WavevectorInfo::WavevectorInfo "WavevectorInfo::WavevectorInfo(kvector_t ki, kvector_t kf, double wavelength)
";

%feature("docstring")  WavevectorInfo::transformed "WavevectorInfo WavevectorInfo::transformed(const Transform3D &transform) const 
";

%feature("docstring")  WavevectorInfo::getKi "cvector_t WavevectorInfo::getKi() const 
";

%feature("docstring")  WavevectorInfo::getKf "cvector_t WavevectorInfo::getKf() const 
";

%feature("docstring")  WavevectorInfo::getQ "cvector_t WavevectorInfo::getQ() const 
";

%feature("docstring")  WavevectorInfo::getWavelength "double WavevectorInfo::getWavelength() const 
";


// File: classConvolve_1_1Workspace.xml


// File: namespace_0D147.xml


// File: namespace_0D190.xml


// File: namespace_0D270.xml


// File: namespace_0D282.xml


// File: namespace_0D308.xml


// File: namespace_0D310.xml


// File: namespace_0D323.xml


// File: namespace_0D436.xml


// File: namespace_0D62.xml


// File: namespace_0D77.xml


// File: namespaceArrayUtils.xml
%feature("docstring")  ArrayUtils::getShape "std::pair< size_t, size_t > ArrayUtils::getShape(const T &data)

Returns shape nrows, ncols of 2D array. 
";


// File: namespaceboost_1_1geometry.xml


// File: namespaceBornAgain.xml


// File: namespaceDataFormatUtils.xml
%feature("docstring")  DataFormatUtils::isCompressed "bool DataFormatUtils::isCompressed(const std::string &name)

Returns true if name contains *.gz extension. 
";

%feature("docstring")  DataFormatUtils::isGZipped "bool DataFormatUtils::isGZipped(const std::string &name)

Returns true if name contains *.gz extension.

Does name contain *.gz extension? 
";

%feature("docstring")  DataFormatUtils::isBZipped "bool DataFormatUtils::isBZipped(const std::string &name)

Returns true if name contains *.bz2 extension. 
";

%feature("docstring")  DataFormatUtils::GetFileMainExtension "std::string DataFormatUtils::GetFileMainExtension(const std::string &name)

Returns file extension after stripping '.gz' if any.

Returns file main extension (without .gz). 
";

%feature("docstring")  DataFormatUtils::isBinaryFile "bool DataFormatUtils::isBinaryFile(const std::string &file_name)

returns true if file name corresponds to a binary file 
";

%feature("docstring")  DataFormatUtils::isIntFile "bool DataFormatUtils::isIntFile(const std::string &file_name)

returns true if file name corresponds to  BornAgain native format (compressed or not) 
";

%feature("docstring")  DataFormatUtils::isTxtFile "bool DataFormatUtils::isTxtFile(const std::string &file_name)

returns true if file name corresponds to simple numpy-style ASCII file 
";

%feature("docstring")  DataFormatUtils::isTiffFile "bool DataFormatUtils::isTiffFile(const std::string &file_name)

returns true if file name corresponds to tiff file (can be also compressed) 
";

%feature("docstring")  DataFormatUtils::isSimilarToFixedBinAxisType "bool DataFormatUtils::isSimilarToFixedBinAxisType(const std::string &line)

Returns true if string representation of the axis contains one of  FixedBinAxis,  ConstKBinAxis or  CustomBinAxis to parse it later in similar way. 
";

%feature("docstring")  DataFormatUtils::isVariableBinAxisType "bool DataFormatUtils::isVariableBinAxisType(const std::string &line)
";

%feature("docstring")  DataFormatUtils::createAxis "IAxis * DataFormatUtils::createAxis(std::istream &input_stream)

Creates axis of certain type from input stream. 
";

%feature("docstring")  DataFormatUtils::createFixedBinAxis "IAxis * DataFormatUtils::createFixedBinAxis(std::string line)

Create one of  FixedBinAxis from string representation  FixedBinAxis(\"axis0\", 10, -1, 1)  ConstKBinAxis(\"axis0\", 10, -1, 1)  CustomBinAxis(\"axis0\", 10, -1, 1) 
";

%feature("docstring")  DataFormatUtils::createVariableBinAxis "IAxis * DataFormatUtils::createVariableBinAxis(std::string line)

Create  VariableBinAxis from string representation  VariableBinAxis(\"axis0\", 4, [-1, -0.5, 0.5, 1, 2]) 
";

%feature("docstring")  DataFormatUtils::fillOutputData "void DataFormatUtils::fillOutputData(OutputData< double > *data, std::istream &input_stream)

Fills output data raw buffer from input stream. 
";

%feature("docstring")  DataFormatUtils::parse_doubles "std::vector< double > DataFormatUtils::parse_doubles(const std::string &str)

Parse double values from string to vector of double. 
";

%feature("docstring")  DataFormatUtils::readLineOfDoubles "void DataFormatUtils::readLineOfDoubles(std::vector< double > &buffer, std::istringstream &iss)
";


// File: namespaceDetectorFunctions.xml
%feature("docstring")  DetectorFunctions::hasSameDimensions "bool DetectorFunctions::hasSameDimensions(const IDetector2D &detector, const OutputData< double > &data)

Returns true if the data has same axes size (nx,ny) with the detector. 
";

%feature("docstring")  DetectorFunctions::axesToString "std::string DetectorFunctions::axesToString(const IDetector2D &detector)

Returns string representation of axes dimension in the form \"(nx,ny)\". 
";

%feature("docstring")  DetectorFunctions::axesToString "std::string DetectorFunctions::axesToString(const OutputData< double > &data)

Returns string representation of axes dimension in the form \"(nx,ny)\". 
";

%feature("docstring")  DetectorFunctions::createDataSet "std::unique_ptr< OutputData< double > > DetectorFunctions::createDataSet(const Instrument &instrument, const OutputData< double > &data, bool put_masked_areas_to_zero=true, IDetector2D::EAxesUnits units=IDetector2D::DEFAULT)

Creates real data containing original user data clipped to the ROI area of the detector. If put_masked_areas_to_zero==true: resulting data will have 0.0 in all masked areas If put_masked_areas_to_zero==false: resulting data will be only cropped, masked areas will still contain intensities TODO: what users will like more (this appears on FitSuitePlotObserver)? 
";


// File: namespaceExceptions.xml
%feature("docstring")  Exceptions::LogExceptionMessage "void Exceptions::LogExceptionMessage(const std::string &message)
";


// File: namespaceFileSystemUtils.xml
%feature("docstring")  FileSystemUtils::extension "std::string FileSystemUtils::extension(const std::string &fname)

Returns extension of given filename. 
";

%feature("docstring")  FileSystemUtils::createDirectory "bool FileSystemUtils::createDirectory(const std::string &dir_name)

Creates directory in current directory. 
";

%feature("docstring")  FileSystemUtils::filesInDirectory "std::vector< std::string > FileSystemUtils::filesInDirectory(const std::string &dir_name)

Returns filenames of files in directory. 
";

%feature("docstring")  FileSystemUtils::jointPath "std::string FileSystemUtils::jointPath(const std::string &spath1, const std::string &spath2)

Returns joint path name. 
";

%feature("docstring")  FileSystemUtils::filename "std::string FileSystemUtils::filename(const std::string &path)

Returns path without directory part (\"Foo/Bar/Doz.int.gz\" -> \"Doz.int.gz\") 
";

%feature("docstring")  FileSystemUtils::glob "std::vector< std::string > FileSystemUtils::glob(const std::string &dir, const std::string &pattern)

Returns file names that agree with a regex glob pattern. 
";


// File: namespaceFitSuiteUtils.xml
%feature("docstring")  FitSuiteUtils::linkedParameters "std::vector< FitParameter * > FitSuiteUtils::linkedParameters(const FitParameterSet &fitParameters)

Returns vector of linked parameters. 
";

%feature("docstring")  FitSuiteUtils::linkToString "std::string FitSuiteUtils::linkToString(const FitParameter &par)

Returns multiline string representing linked sample parameters. 
";

%feature("docstring")  FitSuiteUtils::fitParameterSettingsToString "std::string FitSuiteUtils::fitParameterSettingsToString(const FitParameterSet &fitParameters)

Returns multiline string representing fit parameter definition. 
";

%feature("docstring")  FitSuiteUtils::hasConflicts "bool FitSuiteUtils::hasConflicts(const FitParameterSet &fitParameters)

Validates all fit parameters for conflicts (steering same sample parameters). 
";


// File: namespaceIntensityDataFunctions.xml
%feature("docstring")  IntensityDataFunctions::getRelativeDifference "double IntensityDataFunctions::getRelativeDifference(const OutputData< double > &dat, const OutputData< double > &ref)

Returns relative difference between two data sets sum(dat[i] - ref[i])/ref[i]). 
";

%feature("docstring")  IntensityDataFunctions::getRelativeDifference "double IntensityDataFunctions::getRelativeDifference(const IHistogram &dat, const IHistogram &ref)
";

%feature("docstring")  IntensityDataFunctions::createRelativeDifferenceData "OutputData< double > * IntensityDataFunctions::createRelativeDifferenceData(const OutputData< double > &data, const OutputData< double > &reference)
";

%feature("docstring")  IntensityDataFunctions::createClippedDataSet "OutputData< double > * IntensityDataFunctions::createClippedDataSet(const OutputData< double > &origin, double x1, double y1, double x2, double y2)

Returns new IntensityData objects which axes clipped to represent the specified rectangle. 
";

%feature("docstring")  IntensityDataFunctions::applyDetectorResolution "OutputData< double > * IntensityDataFunctions::applyDetectorResolution(const OutputData< double > &origin, const IResolutionFunction2D &resolution_function)

Applies detector resolution function and returns new IntensityData object. 
";

%feature("docstring")  IntensityDataFunctions::coordinateToBinf "double IntensityDataFunctions::coordinateToBinf(double coordinate, const IAxis &axis)

Transforms coordinate on axis into the bin-fraction-coordinate. 
";

%feature("docstring")  IntensityDataFunctions::coordinateFromBinf "double IntensityDataFunctions::coordinateFromBinf(double value, const IAxis &axis)

Transforms bin-fraction-coordinate into axis coordinate. 
";

%feature("docstring")  IntensityDataFunctions::coordinateToBinf "void IntensityDataFunctions::coordinateToBinf(double &x, double &y, const OutputData< double > &data)

Transforms x,y coordinate from  OutputData axes coordinates to bin-fraction-coordinates. 
";

%feature("docstring")  IntensityDataFunctions::coordinateFromBinf "void IntensityDataFunctions::coordinateFromBinf(double &x, double &y, const OutputData< double > &data)

Transforms x,y coordinate from bin-fraction-coordinates to  OutputData's axes coordinates. 
";


// File: namespaceMathFunctions.xml
%feature("docstring")  MathFunctions::Bessel_J0_PowSer "complex_t MathFunctions::Bessel_J0_PowSer(const complex_t z)

Computes complex Bessel function J0(z), using power series and asymptotic expansion.

Computes the complex Bessel function J0(z), using power series and asymptotic expansion.

Forked from unoptimized code at http://www.crbond.com/math.htm, who refers to \"Computation of Special Functions\", Zhang and Jin, John Wiley and Sons, 1996. 
";

%feature("docstring")  MathFunctions::Bessel_J1_PowSer "complex_t MathFunctions::Bessel_J1_PowSer(const complex_t z)

Computes complex Bessel function J0(z), using power series and asymptotic expansion.

Computes the complex Bessel function J1(z), using power series and asymptotic expansion.

Forked from same source as for Bessel_J0_PowSer 
";

%feature("docstring")  MathFunctions::StandardNormal "double MathFunctions::StandardNormal(double x)
";

%feature("docstring")  MathFunctions::Gaussian "double MathFunctions::Gaussian(double x, double average, double std_dev)
";

%feature("docstring")  MathFunctions::IntegratedGaussian "double MathFunctions::IntegratedGaussian(double x, double average, double std_dev)
";

%feature("docstring")  MathFunctions::cot "double MathFunctions::cot(double x)

cotangent function:  $cot(x)\\\\equiv1/tan(x)$
";

%feature("docstring")  MathFunctions::Si "double MathFunctions::Si(double x)

Sine integral function:  $Si(x)\\\\equiv\\\\int_0^x du \\\\sin(u)/u$. 
";

%feature("docstring")  MathFunctions::sinc "double MathFunctions::sinc(double x)

sinc function:  $sinc(x)\\\\equiv\\\\sin(x)/x$
";

%feature("docstring")  MathFunctions::sinc "complex_t MathFunctions::sinc(const complex_t z)

Complex sinc function:  $sinc(x)\\\\equiv\\\\sin(x)/x$. 
";

%feature("docstring")  MathFunctions::tanhc "complex_t MathFunctions::tanhc(const complex_t z)

Complex tanhc function:  $tanhc(x)\\\\equiv\\\\tanh(x)/x$. 
";

%feature("docstring")  MathFunctions::Laue "complex_t MathFunctions::Laue(const complex_t z, size_t N)
";

%feature("docstring")  MathFunctions::Bessel_J0 "double MathFunctions::Bessel_J0(double x)

Bessel function of the first kind and order 0. 
";

%feature("docstring")  MathFunctions::Bessel_J1 "double MathFunctions::Bessel_J1(double x)

Bessel function of the first kind and order 1. 
";

%feature("docstring")  MathFunctions::Bessel_J1c "double MathFunctions::Bessel_J1c(double x)

Bessel function Bessel_J1(x)/x. 
";

%feature("docstring")  MathFunctions::Bessel_J0 "complex_t MathFunctions::Bessel_J0(const complex_t z)

Complex Bessel function of the first kind and order 0. 
";

%feature("docstring")  MathFunctions::Bessel_J1 "complex_t MathFunctions::Bessel_J1(const complex_t z)

Complex Bessel function of the first kind and order 1. 
";

%feature("docstring")  MathFunctions::Bessel_J1c "complex_t MathFunctions::Bessel_J1c(const complex_t z)

Complex Bessel function Bessel_J1(x)/x. 
";

%feature("docstring")  MathFunctions::FastFourierTransform "std::vector< complex_t > MathFunctions::FastFourierTransform(const std::vector< complex_t > &data, EFFTDirection tcase)

simple (and unoptimized) wrapper function for the discrete fast Fourier transformation library (fftw3) 
";

%feature("docstring")  MathFunctions::FastFourierTransform "std::vector< complex_t > MathFunctions::FastFourierTransform(const std::vector< double > &data, EFFTDirection tcase)

simple (and unoptimized) wrapper function for the discrete fast Fourier transformation library (fftw3); transforms real to complex 
";

%feature("docstring")  MathFunctions::ConvolveFFT "std::vector< complex_t > MathFunctions::ConvolveFFT(const std::vector< double > &signal, const std::vector< double > &resfunc)

convolution of two real vectors of equal size 
";

%feature("docstring")  MathFunctions::GenerateUniformRandom "double MathFunctions::GenerateUniformRandom()
";

%feature("docstring")  MathFunctions::GenerateStandardNormalRandom "double MathFunctions::GenerateStandardNormalRandom()
";

%feature("docstring")  MathFunctions::GenerateNormalRandom "double MathFunctions::GenerateNormalRandom(double average, double std_dev)
";


// File: namespaceNodeUtils.xml
%feature("docstring")  NodeUtils::nodeToString "std::string NodeUtils::nodeToString(const INode &node)

Returns multiline string representing tree structure starting from given node. 
";

%feature("docstring")  NodeUtils::nodePath "std::string NodeUtils::nodePath(const INode &node, const INode *root=nullptr)

Returns path composed of node's displayName, with respect to root node. 
";


// File: namespaceNumeric.xml
%feature("docstring")  Numeric::areAlmostEqual "bool BA_CORE_API_ Numeric::areAlmostEqual(double a, double b, double tolerance)

Returns true if two doubles agree within epsilon*tolerance. 
";

%feature("docstring")  Numeric::get_relative_difference "double BA_CORE_API_ Numeric::get_relative_difference(double a, double b)

Returns the safe relative difference, which is |(a-b)/b| except in special cases. 
";


// File: namespacePythonFormatting.xml
%feature("docstring")  PythonFormatting::representShape2D "BA_CORE_API_ std::string PythonFormatting::representShape2D(const std::string &indent, const IShape2D *ishape, bool mask_value, std::function< std::string(double)> printValueFunc)

Returns fixed Python code snippet that defines the function \"runSimulation\". 
";

%feature("docstring")  PythonFormatting::printBool "BA_CORE_API_ std::string PythonFormatting::printBool(double value)
";

%feature("docstring")  PythonFormatting::printDouble "BA_CORE_API_ std::string PythonFormatting::printDouble(double input)
";

%feature("docstring")  PythonFormatting::printNm "BA_CORE_API_ std::string PythonFormatting::printNm(double input)
";

%feature("docstring")  PythonFormatting::printScientificDouble "BA_CORE_API_ std::string PythonFormatting::printScientificDouble(double input)
";

%feature("docstring")  PythonFormatting::printDegrees "BA_CORE_API_ std::string PythonFormatting::printDegrees(double input)
";

%feature("docstring")  PythonFormatting::isSquare "BA_CORE_API_ bool PythonFormatting::isSquare(double length1, double length2, double angle)
";

%feature("docstring")  PythonFormatting::isHexagonal "BA_CORE_API_ bool PythonFormatting::isHexagonal(double length1, double length2, double angle)
";

%feature("docstring")  PythonFormatting::printKvector "BA_CORE_API_ std::string PythonFormatting::printKvector(const kvector_t value)
";

%feature("docstring")  PythonFormatting::isDefaultDirection "BA_CORE_API_ bool PythonFormatting::isDefaultDirection(const kvector_t direction)

returns true if it is (0, -1, 0) vector 
";

%feature("docstring")  PythonFormatting::valueTimesUnit "BA_CORE_API_ std::string PythonFormatting::valueTimesUnit(const RealParameter *par)

Returns parameter value, followed by its unit multiplicator (like \"* nm\"). 
";

%feature("docstring")  PythonFormatting::argumentList "BA_CORE_API_ std::string PythonFormatting::argumentList(const IParameterized *ip)

Returns comma-separated list of parameter values, including unit multiplicator (like \"* nm\"). 
";

%feature("docstring")  PythonFormatting::simulationToPython "std::string PythonFormatting::simulationToPython(GISASSimulation *simulation)
";


// File: namespaceStandardSimulations.xml
%feature("docstring")  StandardSimulations::PolarizedDWBAMagCylinders2 "GISASSimulation * StandardSimulations::PolarizedDWBAMagCylinders2()
";

%feature("docstring")  StandardSimulations::BasicGISAS "GISASSimulation * StandardSimulations::BasicGISAS()

Basic GISAS simulation with the detector phi[0,2], theta[0,2]. 
";

%feature("docstring")  StandardSimulations::BasicGISAS00 "GISASSimulation * StandardSimulations::BasicGISAS00()

Basic GISAS for polarization studies. 
";

%feature("docstring")  StandardSimulations::MiniGISAS "GISASSimulation * StandardSimulations::MiniGISAS()

GISAS simulation with small detector and phi[-2,2], theta[0,2]. 
";

%feature("docstring")  StandardSimulations::MiniGISAS_v2 "GISASSimulation * StandardSimulations::MiniGISAS_v2()

GISAS simulation with small detector and phi[-1,1], theta[0,1]. 
";

%feature("docstring")  StandardSimulations::MiniGISASBeamDivergence "GISASSimulation * StandardSimulations::MiniGISASBeamDivergence()

GISAS simulation with beam divergence applied. 
";

%feature("docstring")  StandardSimulations::MiniGISASDetectorResolution "GISASSimulation * StandardSimulations::MiniGISASDetectorResolution()

GISAS simulation with detector resolution. 
";

%feature("docstring")  StandardSimulations::MiniGISASPolarizationPP "GISASSimulation * StandardSimulations::MiniGISASPolarizationPP()
";

%feature("docstring")  StandardSimulations::GISASWithMasks "GISASSimulation * StandardSimulations::GISASWithMasks()

GISAS simulation with multiple masks on the detector plane. 
";

%feature("docstring")  StandardSimulations::MaxiGISAS "GISASSimulation * StandardSimulations::MaxiGISAS()

GISAS simulation with large detector to test performance. 
";

%feature("docstring")  StandardSimulations::MaxiGISAS00 "GISASSimulation * StandardSimulations::MaxiGISAS00()

Basic GISAS for polarization studies. 
";

%feature("docstring")  StandardSimulations::IsGISAXSSimulation1 "GISASSimulation * StandardSimulations::IsGISAXSSimulation1()

Typical IsGISAXS simulation with the detector phi[-1,1], theta[0,2]. 
";

%feature("docstring")  StandardSimulations::IsGISAXSSimulation2 "GISASSimulation * StandardSimulations::IsGISAXSSimulation2()

Typical IsGISAXS simulation with the detector phi[0,2], theta[0,2]. 
";

%feature("docstring")  StandardSimulations::RectDetectorGeneric "GISASSimulation * StandardSimulations::RectDetectorGeneric()

GISAS simulation with generic rectangular detector. 
";

%feature("docstring")  StandardSimulations::RectDetectorPerpToSample "GISASSimulation * StandardSimulations::RectDetectorPerpToSample()

GISAS simulation with the rectangular detector perpendicular to the sample. 
";

%feature("docstring")  StandardSimulations::RectDetectorPerpToDirectBeam "GISASSimulation * StandardSimulations::RectDetectorPerpToDirectBeam()

GISAS simulation with the rectangular detector perpendicular to the direct beam. 
";

%feature("docstring")  StandardSimulations::RectDetectorPerpToReflectedBeam "GISASSimulation * StandardSimulations::RectDetectorPerpToReflectedBeam()

GISAS simulation with the rectangular detector perpendicular to the reflected beam. 
";

%feature("docstring")  StandardSimulations::RectDetectorPerpToReflectedBeamDpos "GISASSimulation * StandardSimulations::RectDetectorPerpToReflectedBeamDpos()

GISAS simulation with the rectangular detector perpendicular to the reflected beam when the coordinates of direct beam are known. 
";

%feature("docstring")  StandardSimulations::MiniGISASMonteCarlo "GISASSimulation * StandardSimulations::MiniGISASMonteCarlo()

GISAS simulation with Monte-Carlo integration switched ON. 
";

%feature("docstring")  StandardSimulations::SphericalDetWithRoi "GISASSimulation * StandardSimulations::SphericalDetWithRoi()

GISAS simulation with spherical detector, region of interest and mask. 
";

%feature("docstring")  StandardSimulations::RectDetWithRoi "GISASSimulation * StandardSimulations::RectDetWithRoi()

GISAS simulation with rectangular detector, region of interest and mask. 
";


// File: namespaceSysUtils.xml
%feature("docstring")  SysUtils::getCurrentDateAndTime "std::string SysUtils::getCurrentDateAndTime()
";

%feature("docstring")  SysUtils::enableFloatingPointExceptions "void SysUtils::enableFloatingPointExceptions()

enables exception throw in the case of NaN, Inf 
";


// File: namespaceUnits.xml
%feature("docstring")  Units::rad2deg "double Units::rad2deg(double angle)
";

%feature("docstring")  Units::deg2rad "double Units::deg2rad(double angle)
";


// File: FormFactorCoherentPart_8cpp.xml


// File: FormFactorCoherentPart_8h.xml


// File: FormFactorCoherentSum_8cpp.xml


// File: FormFactorCoherentSum_8h.xml


// File: FTDecayFunctions_8cpp.xml


// File: FTDecayFunctions_8h.xml


// File: FTDistributions1D_8cpp.xml


// File: FTDistributions1D_8h.xml


// File: FTDistributions2D_8cpp.xml


// File: FTDistributions2D_8h.xml


// File: IInterferenceFunction_8h.xml


// File: ILayout_8h.xml


// File: InterferenceFunction1DLattice_8cpp.xml


// File: InterferenceFunction1DLattice_8h.xml


// File: InterferenceFunction2DLattice_8cpp.xml


// File: InterferenceFunction2DLattice_8h.xml


// File: InterferenceFunction2DParaCrystal_8cpp.xml


// File: InterferenceFunction2DParaCrystal_8h.xml


// File: InterferenceFunctionNone_8cpp.xml


// File: InterferenceFunctionNone_8h.xml


// File: InterferenceFunctionRadialParaCrystal_8cpp.xml


// File: InterferenceFunctionRadialParaCrystal_8h.xml


// File: ParticleLayout_8cpp.xml


// File: ParticleLayout_8h.xml


// File: BornAgainNamespace_8h.xml


// File: Complex_8h.xml
%feature("docstring")  mul_I "complex_t mul_I(complex_t z)

Returns product I*z, where I is the imaginary unit. 
";

%feature("docstring")  exp_I "complex_t exp_I(complex_t z)

Returns exp(I*z), where I is the imaginary unit. 
";


// File: Exceptions_8cpp.xml


// File: Exceptions_8h.xml


// File: ICloneable_8h.xml


// File: INamed_8h.xml


// File: INoncopyable_8h.xml


// File: ISingleton_8h.xml


// File: Macros_8h.xml


// File: MathConstants_8h.xml


// File: Bin_8cpp.xml
%feature("docstring")  BinContains "bool BinContains(const Bin1D &bin, double value)

Checks if value is contained in bin: value in [m_lower, m_upper) 
";


// File: Bin_8h.xml
%feature("docstring")  BinContains "bool BinContains(const Bin1D &bin, double value)

Checks if value is contained in bin: value in [m_lower, m_upper) 
";


// File: ConstKBinAxis_8cpp.xml


// File: ConstKBinAxis_8h.xml


// File: CustomBinAxis_8cpp.xml


// File: CustomBinAxis_8h.xml


// File: FixedBinAxis_8cpp.xml


// File: FixedBinAxis_8h.xml


// File: IAxis_8cpp.xml


// File: IAxis_8h.xml
%feature("docstring")  HaveSameNameAndShape "bool HaveSameNameAndShape(const IAxis &left, const IAxis &right)

global helper function for comparison of axes 
";


// File: IPixel_8h.xml


// File: SimulationElement_8cpp.xml
%feature("docstring")  addElementsWithWeight "void addElementsWithWeight(std::vector< SimulationElement >::const_iterator first, std::vector< SimulationElement >::const_iterator last, std::vector< SimulationElement >::iterator result, double weight)

Add element vector to element vector with weight. 
";


// File: SimulationElement_8h.xml
%feature("docstring")  addElementsWithWeight "void addElementsWithWeight(std::vector< SimulationElement >::const_iterator first, std::vector< SimulationElement >::const_iterator last, std::vector< SimulationElement >::iterator result, double weight)

Add element vector to element vector with weight. 
";


// File: VariableBinAxis_8cpp.xml


// File: VariableBinAxis_8h.xml


// File: ComputationStatus_8h.xml


// File: DelayedProgressCounter_8cpp.xml


// File: DelayedProgressCounter_8h.xml


// File: FullFresnelMap_8cpp.xml


// File: FullFresnelMap_8h.xml


// File: IComputationTerm_8cpp.xml


// File: IComputationTerm_8h.xml


// File: MainComputation_8cpp.xml


// File: MainComputation_8h.xml


// File: ParticleLayoutComputation_8cpp.xml


// File: ParticleLayoutComputation_8h.xml


// File: ProgressHandler_8cpp.xml


// File: ProgressHandler_8h.xml


// File: RoughMultiLayerComputation_8cpp.xml


// File: RoughMultiLayerComputation_8h.xml


// File: Slice_8h.xml


// File: SpecularComputation_8cpp.xml


// File: SpecularComputation_8h.xml


// File: FormFactorDecoratorDebyeWaller_8cpp.xml


// File: FormFactorDecoratorDebyeWaller_8h.xml


// File: FormFactorDecoratorFactor_8h.xml


// File: FormFactorDecoratorMaterial_8cpp.xml


// File: FormFactorDecoratorMaterial_8h.xml


// File: FormFactorDecoratorPositionFactor_8cpp.xml


// File: FormFactorDecoratorPositionFactor_8h.xml


// File: FormFactorDecoratorRotation_8cpp.xml


// File: FormFactorDecoratorRotation_8h.xml


// File: IFormFactorDecorator_8h.xml


// File: ExportToPython_8cpp.xml


// File: ExportToPython_8h.xml


// File: PythonFormatting_8cpp.xml


// File: PythonFormatting_8h.xml


// File: SampleLabelHandler_8cpp.xml


// File: SampleLabelHandler_8h.xml


// File: AdjustMinimizerStrategy_8cpp.xml


// File: AdjustMinimizerStrategy_8h.xml


// File: FitObject_8cpp.xml


// File: FitObject_8h.xml


// File: FitParameter_8cpp.xml


// File: FitParameter_8h.xml


// File: FitStrategyAdjustParameters_8cpp.xml


// File: FitStrategyAdjustParameters_8h.xml


// File: FitSuite_8cpp.xml


// File: FitSuite_8h.xml


// File: FitSuiteFunctions_8cpp.xml


// File: FitSuiteFunctions_8h.xml


// File: FitSuiteImpl_8cpp.xml


// File: FitSuiteImpl_8h.xml


// File: FitSuiteObjects_8cpp.xml


// File: FitSuiteObjects_8h.xml


// File: FitSuitePrintObserver_8cpp.xml


// File: FitSuitePrintObserver_8h.xml


// File: FitSuiteStrategies_8cpp.xml


// File: FitSuiteStrategies_8h.xml


// File: FitSuiteUtils_8cpp.xml


// File: FitSuiteUtils_8h.xml


// File: IFitObserver_8cpp.xml


// File: IFitObserver_8h.xml


// File: IFitStrategy_8cpp.xml


// File: IFitStrategy_8h.xml


// File: IObserver_8cpp.xml


// File: IObserver_8h.xml


// File: FormFactorAnisoPyramid_8cpp.xml


// File: FormFactorAnisoPyramid_8h.xml


// File: FormFactorBox_8cpp.xml


// File: FormFactorBox_8h.xml


// File: FormFactorCone_8cpp.xml


// File: FormFactorCone_8h.xml


// File: FormFactorCone6_8cpp.xml


// File: FormFactorCone6_8h.xml


// File: FormFactorCuboctahedron_8cpp.xml


// File: FormFactorCuboctahedron_8h.xml


// File: FormFactorCylinder_8cpp.xml


// File: FormFactorCylinder_8h.xml


// File: FormFactorDodecahedron_8cpp.xml


// File: FormFactorDodecahedron_8h.xml


// File: FormFactorEllipsoidalCylinder_8cpp.xml


// File: FormFactorEllipsoidalCylinder_8h.xml


// File: FormFactorFullSphere_8cpp.xml


// File: FormFactorFullSphere_8h.xml


// File: FormFactorFullSpheroid_8cpp.xml


// File: FormFactorFullSpheroid_8h.xml


// File: FormFactorHemiEllipsoid_8cpp.xml


// File: FormFactorHemiEllipsoid_8h.xml


// File: FormFactorIcosahedron_8cpp.xml


// File: FormFactorIcosahedron_8h.xml


// File: FormFactorLongBoxGauss_8cpp.xml


// File: FormFactorLongBoxGauss_8h.xml


// File: FormFactorLongBoxLorentz_8cpp.xml


// File: FormFactorLongBoxLorentz_8h.xml


// File: FormFactorLongRipple1Gauss_8cpp.xml


// File: FormFactorLongRipple1Gauss_8h.xml


// File: FormFactorLongRipple1Lorentz_8cpp.xml


// File: FormFactorLongRipple1Lorentz_8h.xml


// File: FormFactorLongRipple2Gauss_8cpp.xml


// File: FormFactorLongRipple2Gauss_8h.xml


// File: FormFactorLongRipple2Lorentz_8cpp.xml


// File: FormFactorLongRipple2Lorentz_8h.xml


// File: FormFactorPolyhedron_8cpp.xml


// File: FormFactorPolyhedron_8h.xml


// File: FormFactorPrism3_8cpp.xml


// File: FormFactorPrism3_8h.xml


// File: FormFactorPrism6_8cpp.xml


// File: FormFactorPrism6_8h.xml


// File: FormFactorPyramid_8cpp.xml


// File: FormFactorPyramid_8h.xml


// File: FormFactorRipple1_8cpp.xml


// File: FormFactorRipple1_8h.xml


// File: FormFactorRipple2_8cpp.xml


// File: FormFactorRipple2_8h.xml


// File: FormFactorTetrahedron_8cpp.xml


// File: FormFactorTetrahedron_8h.xml


// File: FormFactorTriangle_8cpp.xml


// File: FormFactorTriangle_8h.xml


// File: FormFactorTrivial_8cpp.xml


// File: FormFactorTrivial_8h.xml


// File: FormFactorTruncatedCube_8cpp.xml


// File: FormFactorTruncatedCube_8h.xml


// File: FormFactorTruncatedSphere_8cpp.xml


// File: FormFactorTruncatedSphere_8h.xml


// File: FormFactorTruncatedSpheroid_8cpp.xml


// File: FormFactorTruncatedSpheroid_8h.xml


// File: FormFactors_8h.xml


// File: HardParticles_8h.xml


// File: InterferenceFunctions_8h.xml


// File: SoftParticles_8h.xml


// File: boost__streams_8h.xml


// File: DataFormatUtils_8cpp.xml


// File: DataFormatUtils_8h.xml


// File: IntensityDataIOFactory_8cpp.xml


// File: IntensityDataIOFactory_8h.xml


// File: OutputDataReader_8cpp.xml


// File: OutputDataReader_8h.xml


// File: OutputDataReadFactory_8cpp.xml


// File: OutputDataReadFactory_8h.xml


// File: OutputDataReadStrategy_8cpp.xml


// File: OutputDataReadStrategy_8h.xml


// File: OutputDataWriteFactory_8cpp.xml


// File: OutputDataWriteFactory_8h.xml


// File: OutputDataWriter_8cpp.xml


// File: OutputDataWriter_8h.xml


// File: OutputDataWriteStrategy_8cpp.xml


// File: OutputDataWriteStrategy_8h.xml


// File: TiffHandler_8cpp.xml


// File: TiffHandler_8h.xml


// File: ArrayUtils_8cpp.xml


// File: ArrayUtils_8h.xml


// File: Beam_8cpp.xml


// File: Beam_8h.xml


// File: ChiSquaredModule_8cpp.xml


// File: ChiSquaredModule_8h.xml


// File: ConvolutionDetectorResolution_8cpp.xml


// File: ConvolutionDetectorResolution_8h.xml


// File: Convolve_8cpp.xml


// File: Convolve_8h.xml


// File: CumulativeValue_8cpp.xml


// File: CumulativeValue_8h.xml


// File: DetectionProperties_8cpp.xml


// File: DetectionProperties_8h.xml


// File: DetectorFunctions_8cpp.xml


// File: DetectorFunctions_8h.xml


// File: DetectorMask_8cpp.xml


// File: DetectorMask_8h.xml


// File: FitElement_8cpp.xml


// File: FitElement_8h.xml


// File: Histogram1D_8cpp.xml


// File: Histogram1D_8h.xml


// File: Histogram2D_8cpp.xml


// File: Histogram2D_8h.xml


// File: IChiSquaredModule_8cpp.xml


// File: IChiSquaredModule_8h.xml


// File: IDetector2D_8cpp.xml


// File: IDetector2D_8h.xml


// File: IDetectorResolution_8h.xml


// File: IHistogram_8cpp.xml


// File: IHistogram_8h.xml


// File: IIntensityFunction_8cpp.xml


// File: IIntensityFunction_8h.xml


// File: IIntensityNormalizer_8cpp.xml


// File: IIntensityNormalizer_8h.xml


// File: Instrument_8cpp.xml


// File: Instrument_8h.xml


// File: IntensityDataFunctions_8cpp.xml


// File: IntensityDataFunctions_8h.xml


// File: IResolutionFunction2D_8h.xml


// File: IsGISAXSDetector_8cpp.xml


// File: IsGISAXSDetector_8h.xml


// File: ISquaredFunction_8h.xml


// File: LLData_8cpp.xml


// File: LLData_8h.xml
%feature("docstring")  HaveSameDimensions "bool HaveSameDimensions(const LLData< T > &left, const LLData< T > &right)
";


// File: OutputData_8cpp.xml


// File: OutputData_8h.xml


// File: OutputDataIterator_8h.xml
%feature("docstring")  swap "void swap(OutputDataIterator< TValue, TContainer > &left, OutputDataIterator< TValue, TContainer > &right)

make Swappable 
";


// File: RectangularDetector_8cpp.xml


// File: RectangularDetector_8h.xml


// File: RegionOfInterest_8cpp.xml


// File: RegionOfInterest_8h.xml


// File: ResolutionFunction2DGaussian_8cpp.xml


// File: ResolutionFunction2DGaussian_8h.xml


// File: SimulationArea_8cpp.xml


// File: SimulationArea_8h.xml


// File: SimulationAreaIterator_8cpp.xml


// File: SimulationAreaIterator_8h.xml


// File: SphericalDetector_8cpp.xml


// File: SphericalDetector_8h.xml


// File: ISelectionRule_8h.xml


// File: Lattice_8cpp.xml


// File: Lattice_8h.xml


// File: Lattice1DParameters_8h.xml


// File: Lattice2D_8cpp.xml


// File: Lattice2D_8h.xml


// File: Ellipse_8cpp.xml


// File: Ellipse_8h.xml


// File: InfinitePlane_8h.xml


// File: IShape2D_8h.xml


// File: Line_8cpp.xml


// File: Line_8h.xml


// File: Polygon_8cpp.xml


// File: Polygon_8h.xml


// File: Rectangle_8cpp.xml


// File: Rectangle_8h.xml


// File: HomogeneousMagneticMaterial_8cpp.xml


// File: HomogeneousMagneticMaterial_8h.xml


// File: HomogeneousMaterial_8cpp.xml


// File: HomogeneousMaterial_8h.xml


// File: IMaterial_8cpp.xml


// File: IMaterial_8h.xml


// File: DecouplingApproximationStrategy_8cpp.xml


// File: DecouplingApproximationStrategy_8h.xml


// File: DWBADiffuseReflection_8cpp.xml


// File: DWBADiffuseReflection_8h.xml


// File: FormFactorDWBA_8cpp.xml


// File: FormFactorDWBA_8h.xml


// File: FormFactorDWBAPol_8cpp.xml


// File: FormFactorDWBAPol_8h.xml


// File: IInterferenceFunctionStrategy_8cpp.xml


// File: IInterferenceFunctionStrategy_8h.xml


// File: ILayerRTCoefficients_8h.xml


// File: ILayerSpecularInfo_8cpp.xml


// File: ILayerSpecularInfo_8h.xml


// File: IMultiLayerBuilder_8cpp.xml


// File: IMultiLayerBuilder_8h.xml


// File: Layer_8cpp.xml


// File: Layer_8h.xml


// File: LayerInterface_8cpp.xml


// File: LayerInterface_8h.xml


// File: LayerRoughness_8cpp.xml


// File: LayerRoughness_8h.xml


// File: LayerStrategyBuilder_8cpp.xml


// File: LayerStrategyBuilder_8h.xml


// File: MatrixRTCoefficients_8cpp.xml


// File: MatrixRTCoefficients_8h.xml


// File: MatrixSpecularInfoMap_8cpp.xml


// File: MatrixSpecularInfoMap_8h.xml


// File: MultiLayer_8cpp.xml


// File: MultiLayer_8h.xml


// File: ScalarRTCoefficients_8h.xml


// File: ScalarSpecularInfoMap_8cpp.xml


// File: ScalarSpecularInfoMap_8h.xml


// File: SpecularMagnetic_8cpp.xml


// File: SpecularMagnetic_8h.xml


// File: SpecularMatrix_8cpp.xml
%feature("docstring")  setZeroBelow "void setZeroBelow(SpecularMatrix::MultiLayerCoeff_t &coeff, size_t current_layer)

Sets coeff to zero for all layers below current_layer. 
";

%feature("docstring")  calculateUpFromLayer "bool calculateUpFromLayer(SpecularMatrix::MultiLayerCoeff_t &coeff, const MultiLayer &sample, const double kmag, size_t layer_index)

Computes RT coefficients coeff, starting from layer number layer_index. Returns true if no overflow happens. 
";

%feature("docstring")  bisectRTcomputation "size_t bisectRTcomputation(SpecularMatrix::MultiLayerCoeff_t &coeff, const MultiLayer &sample, const double kmag, const size_t lgood, const size_t lbad, const size_t l)

Recursive bisection to determine the number of the deepest layer where RT computation can be started without running into overflow. Computes coeff, and returns largest possible start layer index. 
";


// File: SpecularMatrix_8h.xml


// File: SSCApproximationStrategy_8cpp.xml


// File: SSCApproximationStrategy_8h.xml


// File: DistributionHandler_8cpp.xml


// File: DistributionHandler_8h.xml


// File: Distributions_8cpp.xml


// File: Distributions_8h.xml


// File: INode_8cpp.xml


// File: INode_8h.xml


// File: IParameter_8h.xml


// File: IParameterized_8cpp.xml


// File: IParameterized_8h.xml


// File: NodeUtils_8cpp.xml


// File: NodeUtils_8h.xml


// File: ParameterDistribution_8cpp.xml


// File: ParameterDistribution_8h.xml


// File: ParameterPattern_8cpp.xml


// File: ParameterPattern_8h.xml


// File: ParameterPool_8cpp.xml


// File: ParameterPool_8h.xml


// File: ParameterSample_8h.xml


// File: RealParameter_8cpp.xml


// File: RealParameter_8h.xml


// File: SimulationOptions_8cpp.xml


// File: SimulationOptions_8h.xml


// File: ThreadInfo_8h.xml


// File: Unit_8cpp.xml


// File: Unit_8h.xml


// File: Units_8h.xml


// File: Crystal_8cpp.xml


// File: Crystal_8h.xml


// File: FormFactorCrystal_8cpp.xml


// File: FormFactorCrystal_8h.xml


// File: FormFactorWeighted_8cpp.xml


// File: FormFactorWeighted_8h.xml


// File: IAbstractParticle_8h.xml


// File: IClusteredParticles_8h.xml


// File: IParticle_8cpp.xml


// File: IParticle_8h.xml


// File: MesoCrystal_8cpp.xml


// File: MesoCrystal_8h.xml


// File: Particle_8cpp.xml


// File: Particle_8h.xml


// File: ParticleComposition_8cpp.xml


// File: ParticleComposition_8h.xml


// File: ParticleCoreShell_8cpp.xml


// File: ParticleCoreShell_8h.xml


// File: ParticleDistribution_8cpp.xml


// File: ParticleDistribution_8h.xml


// File: TRange_8h.xml


// File: IFormFactor_8cpp.xml


// File: IFormFactor_8h.xml


// File: IFormFactorBorn_8cpp.xml


// File: IFormFactorBorn_8h.xml


// File: INodeVisitor_8cpp.xml
%feature("docstring")  VisitSampleTreePreorder "void VisitSampleTreePreorder(const INode &node, INodeVisitor &visitor)
";

%feature("docstring")  VisitSampleTreePostorder "void VisitSampleTreePostorder(const INode &node, INodeVisitor &visitor)
";


// File: INodeVisitor_8h.xml
%feature("docstring")  VisitSampleTreePreorder "BA_CORE_API_ void VisitSampleTreePreorder(const INode &node, INodeVisitor &visitor)
";

%feature("docstring")  VisitSampleTreePostorder "BA_CORE_API_ void VisitSampleTreePostorder(const INode &node, INodeVisitor &visitor)
";


// File: ISample_8cpp.xml


// File: ISample_8h.xml


// File: ISampleIteratorStrategy_8cpp.xml


// File: ISampleIteratorStrategy_8h.xml


// File: Rotations_8cpp.xml
%feature("docstring")  createProduct "IRotation* createProduct(const IRotation &left, const IRotation &right)

Returns concatenated rotation (first right, then left). 
";


// File: Rotations_8h.xml
%feature("docstring")  createProduct "BA_CORE_API_ IRotation* createProduct(const IRotation &left, const IRotation &right)

Returns concatenated rotation (first right, then left). 
";


// File: SampleTreeIterator_8cpp.xml


// File: SampleTreeIterator_8h.xml


// File: GISASSimulation_8cpp.xml


// File: GISASSimulation_8h.xml


// File: OffSpecSimulation_8cpp.xml


// File: OffSpecSimulation_8h.xml


// File: Simulation_8cpp.xml


// File: Simulation_8h.xml


// File: SpecularSimulation_8cpp.xml


// File: SpecularSimulation_8h.xml


// File: FormFactorGauss_8cpp.xml


// File: FormFactorGauss_8h.xml


// File: FormFactorLorentz_8cpp.xml


// File: FormFactorLorentz_8h.xml


// File: FormFactorSphereGaussianRadius_8cpp.xml


// File: FormFactorSphereGaussianRadius_8h.xml


// File: FormFactorSphereLogNormalRadius_8cpp.xml


// File: FormFactorSphereLogNormalRadius_8h.xml


// File: FormFactorSphereUniformRadius_8cpp.xml


// File: FormFactorSphereUniformRadius_8h.xml


// File: BoxCompositionBuilder_8cpp.xml


// File: BoxCompositionBuilder_8h.xml


// File: CoreShellParticleBuilder_8cpp.xml


// File: CoreShellParticleBuilder_8h.xml


// File: CustomMorphologyBuilder_8cpp.xml


// File: CustomMorphologyBuilder_8h.xml


// File: CylindersAndPrismsBuilder_8cpp.xml


// File: CylindersAndPrismsBuilder_8h.xml


// File: CylindersBuilder_8cpp.xml


// File: CylindersBuilder_8h.xml


// File: IFactory_8h.xml
%feature("docstring")  create_new "T* create_new()
";


// File: LatticeBuilder_8cpp.xml


// File: LatticeBuilder_8h.xml


// File: LayersWithAbsorptionBuilder_8cpp.xml


// File: LayersWithAbsorptionBuilder_8h.xml


// File: MagneticLayersBuilder_8cpp.xml


// File: MagneticLayersBuilder_8h.xml


// File: MagneticParticlesBuilder_8cpp.xml


// File: MagneticParticlesBuilder_8h.xml


// File: MesoCrystalBuilder_8cpp.xml


// File: MesoCrystalBuilder_8h.xml


// File: MultiLayerWithRoughnessBuilder_8cpp.xml


// File: MultiLayerWithRoughnessBuilder_8h.xml


// File: MultipleLayoutBuilder_8cpp.xml


// File: MultipleLayoutBuilder_8h.xml


// File: ParaCrystalBuilder_8cpp.xml


// File: ParaCrystalBuilder_8h.xml


// File: ParticleCompositionBuilder_8cpp.xml


// File: ParticleCompositionBuilder_8h.xml


// File: ParticleDistributionsBuilder_8cpp.xml


// File: ParticleDistributionsBuilder_8h.xml


// File: ParticleInTheAirBuilder_8cpp.xml


// File: ParticleInTheAirBuilder_8h.xml


// File: RipplesBuilder_8cpp.xml


// File: RipplesBuilder_8h.xml


// File: RotatedPyramidsBuilder_8cpp.xml


// File: RotatedPyramidsBuilder_8h.xml


// File: SampleBuilderFactory_8cpp.xml


// File: SampleBuilderFactory_8h.xml


// File: SimulationFactory_8cpp.xml


// File: SimulationFactory_8h.xml


// File: SizeDistributionModelsBuilder_8cpp.xml


// File: SizeDistributionModelsBuilder_8h.xml


// File: StandardSimulations_8cpp.xml


// File: StandardSimulations_8h.xml


// File: TransformationsBuilder_8cpp.xml


// File: TransformationsBuilder_8h.xml


// File: TwoDimLatticeBuilder_8cpp.xml


// File: TwoDimLatticeBuilder_8h.xml


// File: FileSystemUtils_8cpp.xml


// File: FileSystemUtils_8h.xml


// File: fp__exception__glibc__extension_8h.xml
%feature("docstring")  fegetexcept "int fegetexcept(void)

License: Public Domain 
David N. Williams 
";

%feature("docstring")  feenableexcept "int feenableexcept(unsigned int excepts)
";

%feature("docstring")  fedisableexcept "int fedisableexcept(unsigned int excepts)
";

%feature("docstring")  fhdl "void fhdl(int sig, siginfo_t *sip, ucontext_t *scp)
";


// File: IntegratorComplex_8h.xml
%feature("docstring")  make_integrator_complex "P_integrator_complex<T> make_integrator_complex(const T *object, complex_integrand< T > mem_function)

Template function to create an integrator object. 
";


// File: IntegratorMCMiser_8h.xml
%feature("docstring")  make_integrator_miser "P_integrator_miser<T> make_integrator_miser(const T *object, miser_integrand< T > mem_function, size_t dim)

Template function to create an integrator object 
";


// File: IntegratorReal_8h.xml
%feature("docstring")  make_integrator_real "P_integrator_real<T> make_integrator_real(const T *object, real_integrand< T > mem_function)

Template function to create an integrator object. 
";


// File: MathFunctions_8cpp.xml


// File: MathFunctions_8h.xml


// File: Numeric_8cpp.xml


// File: Numeric_8h.xml


// File: Precomputed_8cpp.xml


// File: Precomputed_8h.xml


// File: PythonCore_8h.xml


// File: SafePointerVector_8h.xml


// File: SysUtils_8cpp.xml


// File: SysUtils_8h.xml


// File: w32pragma_8h.xml


// File: BasicVector3D_8cpp.xml
%feature("docstring")  vecOfLambdaAlphaPhi "BasicVector3D<double> vecOfLambdaAlphaPhi(const double _lambda, const double _alpha, const double _phi)

Creates a vector<double> as a wavevector with given wavelength and angles. Specifically needed for grazing-incidence scattering. 
";


// File: BasicVector3D_8h.xml
%feature("docstring")  vecOfLambdaAlphaPhi "BA_CORE_API_ BasicVector3D<double> vecOfLambdaAlphaPhi(const double _lambda, const double _alpha, const double _phi)

Creates a vector<double> as a wavevector with given wavelength and angles. Specifically needed for grazing-incidence scattering. 
";


// File: EigenCore_8h.xml


// File: Transform3D_8cpp.xml


// File: Transform3D_8h.xml


// File: Vectors3D_8h.xml


// File: WavevectorInfo_8cpp.xml


// File: WavevectorInfo_8h.xml


// File: dir_52a2c863b7b3435f7dcd40f26828d521.xml


// File: dir_41e08c09ca0aab46c4ada92f12a8c00b.xml


// File: dir_4544cbc948815333bef1258cf6b298b8.xml


// File: dir_d0c8f8fb9032c27878972645c4679f14.xml


// File: dir_404b7d29693a4f046d60c2eccafd1df4.xml


// File: dir_c83916cd1ff49c9e86c8a91c5655951d.xml


// File: dir_59be1faf7048e95263c2fcba140abda1.xml


// File: dir_e746abb3ff095e53619d5a61a48e781a.xml


// File: dir_554fcc4911648c79d524724e80d45fa4.xml


// File: dir_cc3c45a5d33be920aaf94cb9b9fbdb35.xml


// File: dir_9a756f0b2738ef3b5663c172b32b6a4b.xml


// File: dir_e8bc32d0cf85ef86a42504cd31af1370.xml


// File: dir_8b890ad49a09d8f36525f5af93e5737c.xml


// File: dir_9bdb7f774cce5b77ddd3ed60472b168c.xml


// File: dir_74beab5553c7ad06e27a6baadceea9c3.xml


// File: dir_bcc7f66c041cef9b775368068412e104.xml


// File: dir_95667ae48b286f0957284f712e6e3af5.xml


// File: dir_4251a3aefb390b6051267154c2f94d1e.xml


// File: dir_529c0a19338d84aadf389c7b83eb56b1.xml


// File: dir_051c0ff7ebc48614253af3001519ace0.xml


// File: dir_f59c6b3c978505a5ca3672a364c1918e.xml


// File: dir_871fae137308712382f6192f4445a900.xml


// File: dir_44b1a8f39c14c02f6e3c2be419aa97b0.xml


// File: dir_1a0696269c107461a4ce8ff1a48cd2f2.xml


// File: dir_7f288243cf9c204a176dfbf45ea9d349.xml

