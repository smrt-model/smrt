function result = memlsscatt(fGHz,Ti,Wi, roi, Sppt, pci ,sccho, graintype)
%   Scattering coefficient and absorption from MEMLS

%  Aux. input parameter for dielectric model of snow:

%graintype=2;	% 1, 2 or 3 for different assumptions:
%"1" empirical snow measurements, "2" small spheres, "3" thin shells

c0=0.299793;               % vac. speed of light in m/ns

roi = roi./1000; % transforms density to units of g/cm3

cc=vK2eps(roi, graintype);
v =cc(:,1);                  % ice volume fraction
kq=cc(:,2);                  % K^2 ratio
epsid=cc(:,3);               % epsilon of dry snow
nid=sqrt(epsid);             % real refract.index of dry snow
freq=fGHz;
eii=epsaliceimag(freq,Ti,Sppt); % imag epsilon of saline ice
epsiid=v.*eii.*kq.*nid;  % imag epsilon of dry snow component
epsd=epsid+i*epsiid;
eps  = epswet(freq,Ti,Wi,epsd);
epsi =real(eps);         % real epsilon of snow, dry or wet
epsii=imag(eps);         % imag epsilon of snow, dry or wet
gai = (4*pi*freq).*imag(sqrt(eps))./c0;  %absorption coeff (1/m)
[gbih,gbiv,gs6,ga2i] = sccoeff(roi,Ti,pci,freq,epsi,gai,sccho,kq);
result=[gs6, gai];
