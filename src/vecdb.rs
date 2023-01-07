use std::borrow::Borrow;
use std::ops::Deref;
use std::path::PathBuf;
use fmmap::tokio::{AsyncMmapFileExt, AsyncMmapFileMut, AsyncMmapFileMutExt, AsyncMmapFileWriter, AsyncOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Vector Database File
pub struct VecDb {
    mmap: AsyncMmapFileMut,
    pub num_vectors: NumVectors,
    pub num_dimensions: NumDimensions,
    pos: usize
}

#[derive(Default, Debug, Copy, Clone)]
pub struct NumVectors(usize);

#[derive(Default, Debug, Copy, Clone)]
pub struct NumDimensions(usize);

impl VecDb {
    const HEADER_SIZE: usize = 16;

    pub async fn open_write<B: Borrow<PathBuf>>(path: B, num_vectors: NumVectors, num_dimensions: NumDimensions) -> Result<VecDb, fmmap::error::Error> {
        let options = AsyncOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .max_size((num_vectors.0 * 4 * num_dimensions.0 + Self::HEADER_SIZE) as u64)
            .len(num_vectors.0 * 4 * num_dimensions.0 + Self::HEADER_SIZE);

        let mut mmap = AsyncMmapFileMut::open_with_options(path.borrow(), options).await?;
        let mut writer = mmap.writer(0)?;
        writer.write_u32(0).await?; // version
        writer.write_u32(u32::MAX).await?; // padding
        writer.write_u32(num_vectors.0 as u32).await?;
        writer.write_u32(num_dimensions.0 as u32).await?;
        writer.flush().await?;

        Ok(Self {
            mmap,
            num_vectors,
            num_dimensions,
            pos: Self::HEADER_SIZE
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
            pos: Self::HEADER_SIZE
        })
    }

    pub async fn write_vec<V: AsRef<[f32]>>(&mut self, vec: V) -> Result<(), fmmap::error::Error> {
        let vec = vec.as_ref();
        assert_eq!(vec.len(), self.num_dimensions.0);
        let mut writer = self.mmap.writer(self.pos)?;
        for float in vec {
            writer.write_f32(*float).await?;
        }
        self.pos += self.vec_stride();
        Ok(())
    }

    pub async fn read_vec_into<V: AsMut<[f32]>>(&mut self, mut vec: V) -> Result<(), fmmap::error::Error> {
        let vec = vec.as_mut();
        assert_eq!(vec.len(), self.num_dimensions.0);
        let mut reader = self.mmap.reader(self.pos)?;
        for i in  0..self.num_dimensions.0 {
            vec[i] = reader.read_f32().await?;
        }
        self.pos += self.vec_stride();
        Ok(())
    }

    pub async fn read_vec(&mut self) -> Result<Vec<f32>, fmmap::error::Error> {
        let mut reader = self.mmap.reader(self.pos)?;
        let mut vec = Vec::with_capacity(self.num_dimensions.0);
        for _ in  0..self.num_dimensions.0 {
            vec.push(reader.read_f32().await?);
        }
        self.pos += self.vec_stride();
        Ok(vec)
    }

    pub async fn read_all_vecs<F: FnMut(usize, Vec<f32>) -> ()>(&mut self, mut action: F) -> Result<(), fmmap::error::Error> {
        let mut reader = self.mmap.reader(self.pos)?;
        for v in 0..self.num_vectors.0 {
            let mut vec = Vec::with_capacity(self.num_dimensions.0);
            for _ in 0..self.num_dimensions.0 {
                vec.push(reader.read_f32().await?);
            }
            action(v, vec);
            self.pos += self.vec_stride();
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), fmmap::error::Error> {
        self.mmap.flush()?;
        Ok(())
    }

    fn vec_stride(&self) -> usize {
        4 * self.num_dimensions.0
    }
}

impl Drop for VecDb {
    fn drop(&mut self) {
        self.flush().ok();
    }
}

impl From<usize> for NumDimensions {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<usize> for NumVectors {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<u32> for NumDimensions {
    fn from(value: u32) -> Self {
        Self(value as _)
    }
}

impl From<u32> for NumVectors {
    fn from(value: u32) -> Self {
        Self(value as _)
    }
}

impl Into<usize> for NumDimensions {
    fn into(self) -> usize {
        self.0
    }
}

impl Into<usize> for NumVectors {
    fn into(self) -> usize {
        self.0
    }
}

impl AsRef<usize> for NumDimensions {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl AsRef<usize> for NumVectors {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl Deref for NumDimensions {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for NumVectors {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
