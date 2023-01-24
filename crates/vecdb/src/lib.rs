use abstractions::{NumDimensions, NumVectors};
use fmmap::tokio::{AsyncMmapFileExt, AsyncMmapFileMut, AsyncMmapFileMutExt, AsyncOptions};
use std::borrow::Borrow;
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Vector Database File
pub struct VecDb {
    mmap: AsyncMmapFileMut,
    pub num_vectors: NumVectors,
    pub num_dimensions: NumDimensions,
    pos: usize,
}

impl VecDb {
    const HEADER_SIZE: usize = 16;

    pub async fn open_write<B: Borrow<PathBuf>>(
        path: B,
        num_vectors: NumVectors,
        num_dimensions: NumDimensions,
    ) -> Result<VecDb, fmmap::error::Error> {
        let options = AsyncOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .max_size((num_vectors * 4 * num_dimensions + Self::HEADER_SIZE) as u64)
            .len(num_vectors * 4 * num_dimensions + Self::HEADER_SIZE);

        let mut mmap = AsyncMmapFileMut::open_with_options(path.borrow(), options).await?;
        let mut writer = mmap.writer(0)?;
        writer.write_u32(0).await?; // version
        writer.write_u32(u32::MAX).await?; // padding
        writer.write_u32(*num_vectors as u32).await?;
        writer.write_u32(*num_dimensions as u32).await?;
        writer.flush().await?;

        Ok(Self {
            mmap,
            num_vectors,
            num_dimensions,
            pos: Self::HEADER_SIZE,
        })
    }

    pub async fn open_read<B: Borrow<PathBuf>>(path: B) -> Result<VecDb, fmmap::error::Error> {
        let options = AsyncOptions::new()
            .read(true)
            .write(true)
            .create(false)
            .truncate(false);

        let mmap = AsyncMmapFileMut::open_with_options(path.borrow(), options).await?;
        let mut reader = mmap.reader(0)?;
        let version = reader.read_u32().await?;
        assert_eq!(version, 0, "Unsupported file version");
        let _padding = reader.read_u32().await?;
        let num_vectors = reader.read_u32().await?;
        let num_dimensions = reader.read_u32().await?;

        Ok(Self {
            mmap,
            num_vectors: num_vectors.into(),
            num_dimensions: num_dimensions.into(),
            pos: Self::HEADER_SIZE,
        })
    }

    pub async fn write_vec<V: AsRef<[f32]>>(&mut self, vec: V) -> Result<(), std::io::Error> {
        let vec = vec.as_ref();
        assert_eq!(vec.len(), *self.num_dimensions);
        let mut writer = self.mmap.writer(self.pos).unwrap(); // TODO: Fix
        for float in vec {
            writer.write_f32(*float).await?;
        }
        self.pos += self.vec_stride();
        Ok(())
    }

    pub async fn read_vec_into<V: AsMut<[f32]>>(
        &mut self,
        mut vec: V,
    ) -> Result<(), fmmap::error::Error> {
        let vec = vec.as_mut();
        assert_eq!(vec.len(), *self.num_dimensions);
        let mut reader = self.mmap.reader(self.pos)?;
        for i in self.num_dimensions {
            vec[i] = reader.read_f32().await?;
        }
        self.pos += self.vec_stride();
        Ok(())
    }

    pub async fn read_vec(&mut self) -> Result<Vec<f32>, fmmap::error::Error> {
        let mut reader = self.mmap.reader(self.pos)?;
        let mut vec = Vec::with_capacity(*self.num_dimensions);
        for _ in self.num_dimensions {
            vec.push(reader.read_f32().await?);
        }
        self.pos += self.vec_stride();
        Ok(vec)
    }

    /// Reads all vectors from the file.
    /// For each vector, executes the specified function, passing the vector.
    ///
    /// If the provided function returns `true`, the next vector will be processed.
    /// If `false` is returned or no more vectors are available,
    /// processing stops and the number of processed vectors will be returned.
    pub async fn read_all_vecs<F: FnMut(usize, &[f32]) -> bool>(
        &mut self,
        fun: F,
    ) -> Result<usize, fmmap::error::Error> {
        self.read_n_vecs(self.num_vectors, fun).await
    }

    /// Reads all vectors from the file.
    /// For each vector, executes the specified function, passing the vector.
    ///
    /// If the provided function returns `true`, the next vector will be processed.
    /// If `false` is returned or no more vectors are available,
    /// processing stops and the number of processed vectors will be returned.
    pub async fn read_n_vecs<F: FnMut(usize, &[f32]) -> bool>(
        &mut self,
        count: NumVectors,
        mut fun: F,
    ) -> Result<usize, fmmap::error::Error> {
        let count = self.num_vectors.min(count).get();
        let mut reader = self.mmap.reader(self.pos)?;
        let mut vec = vec![0.0; *self.num_dimensions];
        for v in 0..count {
            for i in self.num_dimensions {
                vec[i] = reader.read_f32().await?;
            }
            if !fun(v, &vec) {
                return Ok(v + 1);
            }
            self.pos += self.vec_stride();
        }
        Ok(count)
    }

    pub fn flush(&mut self) -> Result<(), fmmap::error::Error> {
        self.mmap.flush()?;
        Ok(())
    }

    fn vec_stride(&self) -> usize {
        4 * self.num_dimensions
    }
}

impl Drop for VecDb {
    fn drop(&mut self) {
        self.flush().ok();
    }
}
