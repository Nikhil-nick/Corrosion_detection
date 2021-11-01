#--------------------------------------------------------------------------
# THIS PROGRAM SELECT A ROI, QUNATIZE TO LOWER BIT LEVEL AND COMPUTING 
# GRAY LEVEL RUN LENGTH MATRIX AND SEVEN TEXTURE PARAMETERS VIZ., 
#    1. SHORT RUN EMPHASIS (SRE) 
#    2. LONG RUN EMPHASIS(LRE)
#    3. GRAY LEVEL NON-UNIFORMITY (GLN)
#    4. RUN PERCENTAGE (RP)
#    5. RUN LENGTH NON-UNIFORMITY (RLN)
#    6. LOW GRAY LEVEL RUN EMPHASIS (LGRE)
#    7. HIGH GRAY LEVEL RUN EMPHASIS (HGRE)
#--------------------------------------------------------------------------
# AUTHOR: DR. R. KARUNANITHI: DOB 29.12.1974, INDIA
# DATE OF SUBMISSION 15.02.2010
# REF: FRITZ ALBREGSTEN, 1995
#--------------------------------------------------------------------------
#clc(mstring('clear'), mstring('close'), mstring('all'))
im = imread(mstring('file name with path'))
figure()
imshow(im)
im1 = imcrop(im)
im2 = im1(mslice[1:128], mslice[1:128])
im2 = double(im2)
[m, n] = size(im2)
# --------- IMAGE QUANTIZATION TO 4 BITS (16 GRAY LEVELS)------------------
Imax = max(max(im2))
Imin = min(min(im2))
newim = im2 - Imin
Nmax = max(max(newim))
Nmin = min(min(newim))
Q = round(Nmax / 16)
[m, n] = size(newim)
Quant = 0
for i in mslice[1:m]:
    for j in mslice[1:n]:
        I = newim(i, j)
        for B in mslice[1:16]:
            if logical_and((I > Quant), (I <= Quant + Q)):
                newim(i, j).lvalue = B / 16
                Quant = Quant + Q
            end
        end
    end
end
newmax = max(max(newim))
newim1 = newim / newmax
newim2 = round(newim1 * 16) + 1
dir = 0
dist1 = 1
if (dir == 1):
    newim2 = newim2.cT
end
mx = max(max(newim2))
mn = min(min(newim2))
gl = (mx - mn) + 1
[p, q] = size(newim2)
n = p * q
count = 1
c = 1
col = 1
grl(mx, p).lvalue = 0
maxcount(p * q).lvalue = 0
mc = 0
#---------------------COMPUTING GRAY LEVEL RUN LENGTH MATRIX---------------
for j in mslice[1:p]:
    for k in mslice[1:q - dist1]:
        mc = mc + 1
        g = newim2(j, k)
        f = newim2(j, k + dist1)
        if logical_and((g == f), (g != 0)):
            count = count + 1
            c = count
            col = count
            maxcount(mc).lvalue = count
        else:
            grl(g, c).lvalue = grl(g, c) + 1; 
            print (grl)
            col = 1

            count = 1
            c = 1
        end
    end
    grl(f, col).lvalue = grl(f, col) + 1
    count = 1
    c = 1
end
I = (mslice[mx:mn])
m = grl(mslice[mn:mx], mslice[:])
m1 = m.cT
maxrun = max(max(maxcount))
S = 0
G(gl).lvalue = 0
R(q).lvalue = 0
for u in mslice[1:gl]:
    for v in mslice[1:q]:
        G(u).lvalue = G(u) + m(u, v)
        S = S + m(u, v)
    end
end
for u1 in mslice[1:q]:
    for v1 in mslice[1:gl]:
        R(u1).lvalue = R(u1) + m1(u1, v1)
    end
end
[dim, dim1] = size(G)
SRE = 0
LRE = 0 
print LRE
GLN = 0
RLN = 0
RP = 0
LGRE = 0
HGRE = 0

for h1 in mslice[1:maxrun]:
    SRE = SRE + (R(h1) / (h1 * h1))
    LRE = LRE + (R(h1) * (h1 * h1))
    RLN = RLN + (R(h1) * R(h1))
    RP = RP + R(h1)
end
SRE1 = SRE / S
LRE1 = LRE / S
RLN1 = RLN / S
RP1 = RP / n
for h2 in mslice[1:gl]:
    GLN = (GLN + G(h2) ** 2)
    LGRE = LGRE + (G(h2) / (h2 * h2))
    HGRE = HGRE + (h2 * h2) * G(h2)
end
GLN1 = GLN / S
LGRE1 = LGRE / S
HGRE1 = HGRE / S
clc()
# ---------------------------DISPLAY THE PARAMETERS------------------------
disp(sprintf(mstring('%6.4f'), SRE1))
disp(sprintf(mstring('%6.4f'), LRE1))
disp(sprintf(mstring('%6.4f'), GLN1))
disp(sprintf(mstring('%6.4f'), RP1))
disp(sprintf(mstring('%6.4f'), RLN1))
disp(sprintf(mstring('%6.4f'), LGRE1))
disp(sprintf(mstring('%6.4f'), HGRE1))