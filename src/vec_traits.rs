pub trait L2Norm {
    type Output;

    fn l2_norm_sq(&self) -> Self::Output;
    fn l2_norm(&self) -> Self::Output;
}

pub trait Normalize: L2Norm {
    fn normalize_in_place(&mut self);
    fn normalize_into<D: AsMut<[Self::Output]>>(&self, dest: D);
}

impl<T> L2Norm for T
where
    T: AsRef<[f32]>,
{
    type Output = f32;

    fn l2_norm_sq(&self) -> Self::Output {
        self.as_ref().iter().map(|x| x * x).sum()
    }

    fn l2_norm(&self) -> Self::Output {
        match self.l2_norm_sq() {
            0.0 => 1.0,
            x => f32::sqrt(x),
        }
    }
}

impl<T> Normalize for T
where
    T: AsMut<[f32]> + AsRef<[f32]>,
{
    fn normalize_in_place(&mut self) {
        let inv_norm = 1.0 / self.l2_norm();
        for x in self.as_mut().iter_mut() {
            *x *= inv_norm;
        }
    }

    fn normalize_into<D: AsMut<[Self::Output]>>(&self, mut dest: D) {
        let inv_norm = 1.0 / self.l2_norm();
        for (x, y) in self.as_ref().iter().zip(dest.as_mut().iter_mut()) {
            *y = x * inv_norm;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn l2_norm_works() {
        let norm = vec![1.0, 1.0, 0.0].l2_norm();
        assert_relative_eq!(norm, f32::sqrt(2.0), epsilon = 1e-5);
    }

    #[test]
    fn vec_normalize_in_place_works() {
        let mut vec = vec![1.0, 1.0, 0.0];
        vec.normalize_in_place();

        assert_relative_eq!(vec[0], 0.5 * f32::sqrt(2.0), epsilon = 1e-5);
        assert_relative_eq!(vec[1], 0.5 * f32::sqrt(2.0), epsilon = 1e-5);
        assert_eq!(vec[2], 0.0);
    }

    #[test]
    fn slice_normalize_in_place_works() {
        let mut vec = [1.0, 1.0, 0.0];
        vec.normalize_in_place();

        assert_relative_eq!(vec[0], 0.5 * f32::sqrt(2.0), epsilon = 1e-5);
        assert_relative_eq!(vec[1], 0.5 * f32::sqrt(2.0), epsilon = 1e-5);
        assert_eq!(vec[2], 0.0);
    }

    #[test]
    fn vec_normalize_into_works() {
        let vec = vec![1.0, 1.0, 0.0];
        let mut normalized = vec![0.0; 3];
        vec.normalize_into(&mut normalized);

        assert_relative_eq!(normalized[0], 0.5 * f32::sqrt(2.0), epsilon = 1e-5);
        assert_relative_eq!(normalized[1], 0.5 * f32::sqrt(2.0), epsilon = 1e-5);
        assert_eq!(normalized[2], 0.0);
    }
}
