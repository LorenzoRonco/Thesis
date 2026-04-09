"""
Cropping video frames using normalized landmarks for CNN preprocessing.

Questo modulo implementa il ritaglio dell'area di interesse nei video RGB utilizzando
landmarks normalizzati estratti tramite MediaPipe, con smoothing temporale e preprocessing
per MobileNet.

Author: Sign Language Translation Project
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Any
import logging

# Torch è opzionale - fallback su numpy se non disponibile
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoLandmarkCropper:
    """
    Ritaglia l'area di interesse nei video RGB utilizzando landmarks.
    
    Caratteristiche:
    - Denormalizzazione landmarks da pixel originali a coordinate frame
    - Bounding box FISSA su tutto il video (no zoom/dezoom)
    - Padding e aspect ratio quadrato
    - Preprocessing per MobileNet (resize a 200x200 + normalizzazione ImageNet)
    """
    
    # Parametri ImageNet normalization per MobileNet
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def __init__(
        self,
        video_path: str,
        landmarks: np.ndarray,
        padding_percent: float = 0.15,
        target_size: Tuple[int, int] = (200, 200),
        smoothing_method: str = "fixed",
        window_size: int = 5,
    ):
        """
        Inizializza il cropper.
        
        Args:
            video_path: Percorso al video RGB originale
            landmarks: Array di landmarks shape (frames, num_landmarks, 2) in coordinate pixel
            padding_percent: Percentuale di padding attorno alla bounding box (0.15 = 15%)
            target_size: Dimensioni finali del ritaglio (default: 200x200 per MobileNet)
            smoothing_method: "fixed" (bounding box massima fissa, default) | "global" (media) | "moving" (media mobile)
            window_size: Finestra per media mobile (usato se smoothing_method="moving")
        
        Raises:
            ValueError: Se il video non può essere aperto o landmarks ha shape errato
        """
        self.video_path = video_path
        self.padding_percent = padding_percent
        self.target_size = target_size
        self.smoothing_method = smoothing_method
        self.window_size = window_size
        
        # Validate landmarks shape
        if landmarks.ndim != 3 or landmarks.shape[2] != 2:
            raise ValueError(
                f"Landmarks deve avere shape (frames, num_landmarks, 2), "
                f"ricevuto {landmarks.shape}"
            )
        
        if not (0 <= landmarks.min() and landmarks.max() <= 1):
            logger.warning(
                "Landmarks non sono nel range [0, 1]. "
                "Assicurati che siano normalizzati correttamente."
            )
        
        self.landmarks = landmarks
        self.num_frames = landmarks.shape[0]
        
        # Apri il video e ottieni info
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Impossibile aprire il video: {video_path}")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(
            f"Video caricato: {self.frame_width}x{self.frame_height} @ {self.fps} FPS, "
            f"{self.total_frames} frames"
        )
        
        if self.num_frames != self.total_frames:
            logger.warning(
                f"Mismatch: landmarks hanno {self.num_frames} frames, "
                f"video ha {self.total_frames} frames"
            )
    
    def _denormalize_landmarks(self) -> np.ndarray:
        """
        Denormalizza i landmarks da [0,1] a coordinate pixel.
        
        Returns:
            Array di landmarks denormalizzati shape (frames, num_landmarks, 2)
        """
        scale = np.array([self.frame_width, self.frame_height])
        return self.landmarks * scale
    
    def _compute_bounding_boxes_per_frame(
        self,
        denormalized_landmarks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcola x_min, x_max, y_min, y_max per ogni frame.
        
        Args:
            denormalized_landmarks: Array shape (frames, num_landmarks, 2)
        
        Returns:
            Tuple di (x_min, x_max, y_min, y_max), ognuno shape (frames,)
        """
        x_coords = denormalized_landmarks[:, :, 0]  # (frames, num_landmarks)
        y_coords = denormalized_landmarks[:, :, 1]
        
        x_min = np.nanmin(x_coords, axis=1)  # Ignora NaN
        x_max = np.nanmax(x_coords, axis=1)
        y_min = np.nanmin(y_coords, axis=1)
        y_max = np.nanmax(y_coords, axis=1)
        
        return x_min, x_max, y_min, y_max
    
    def _make_square_with_padding(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        y_min: np.ndarray,
        y_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Applica padding e forza la bounding box ad essere quadrata.
        
        Args:
            x_min, x_max, y_min, y_max: Coordinate per ogni frame shape (frames,)
        
        Returns:
            Tuple di coordinate aggiustate (x_min, x_max, y_min, y_max)
        """
        # Calcola centro e dimensioni
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        
        # Dimensione originale della bounding box
        width = x_max - x_min
        height = y_max - y_min
        
        # Prendi il massimo tra larghezza e altezza
        max_size = np.maximum(width, height)
        
        # Applica padding
        box_size = max_size * (1.0 + self.padding_percent)
        
        # Calcola nuove coordinate (centrate)
        x_min_new = x_center - box_size / 2.0
        x_max_new = x_center + box_size / 2.0
        y_min_new = y_center - box_size / 2.0
        y_max_new = y_center + box_size / 2.0
        
        return x_min_new, x_max_new, y_min_new, y_max_new
    
    def _clip_to_frame_bounds(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        y_min: np.ndarray,
        y_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Assicura che le coordinate siano entro i limiti del frame.
        
        Args:
            x_min, x_max, y_min, y_max: Coordinate per ogni frame shape (frames,)
        
        Returns:
            Tuple di coordinate clippate
        """
        x_min = np.clip(x_min, 0, self.frame_width).astype(int)
        x_max = np.clip(x_max, 0, self.frame_width).astype(int)
        y_min = np.clip(y_min, 0, self.frame_height).astype(int)
        y_max = np.clip(y_max, 0, self.frame_height).astype(int)
        
        return x_min, x_max, y_min, y_max
    
    def _apply_temporal_smoothing(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        y_min: np.ndarray,
        y_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Applica smoothing temporale o bounding box fissa.
        
        Args:
            x_min, x_max, y_min, y_max: Coordinate per ogni frame shape (frames,)
        
        Returns:
            Tuple di coordinate smoothate/fisse
        """
        if self.smoothing_method == "fixed":
            # BOUNDING BOX MASSIMA FISSA su tutto il video (no zoom/dezoom)
            x_min_smooth = np.full_like(x_min, np.min(x_min), dtype=np.float32)  # Min più piccolo
            x_max_smooth = np.full_like(x_max, np.max(x_max), dtype=np.float32)  # Max più grande
            y_min_smooth = np.full_like(y_min, np.min(y_min), dtype=np.float32)
            y_max_smooth = np.full_like(y_max, np.max(y_max), dtype=np.float32)
            
            logger.info(
                "Bounding box: FIXED (massima su tutti i frame, no zoom/dezoom) ✓"
            )
        
        elif self.smoothing_method == "global":
            # Usa una bounding box globale (media su tutto il video)
            x_min_smooth = np.full_like(x_min, np.mean(x_min), dtype=np.float32)
            x_max_smooth = np.full_like(x_max, np.mean(x_max), dtype=np.float32)
            y_min_smooth = np.full_like(y_min, np.mean(y_min), dtype=np.float32)
            y_max_smooth = np.full_like(y_max, np.mean(y_max), dtype=np.float32)
            
            logger.info("Bounding box: GLOBAL (media su tutti i frame)")
        
        elif self.smoothing_method == "moving":
            # Media mobile con finestra specificata
            from scipy.ndimage import uniform_filter1d
            
            x_min_smooth = uniform_filter1d(x_min, size=self.window_size, mode='nearest')
            x_max_smooth = uniform_filter1d(x_max, size=self.window_size, mode='nearest')
            y_min_smooth = uniform_filter1d(y_min, size=self.window_size, mode='nearest')
            y_max_smooth = uniform_filter1d(y_max, size=self.window_size, mode='nearest')
            
            logger.info(
                f"Bounding box: MOVING AVERAGE (window_size={self.window_size})"
            )
        
        else:
            raise ValueError(
                f"Metodo smoothing sconosciuto: {self.smoothing_method}. "
                f"Scegli tra 'fixed' (default), 'global' o 'moving'"
            )
        
        return x_min_smooth, x_max_smooth, y_min_smooth, y_max_smooth
    
    def _normalize_for_mobilenet(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalizza il frame secondo ImageNet standard per MobileNet.
        
        Trasformazione:
        1. Converti BGR a RGB e normalizza a [0, 1]
        2. Sottrai media ImageNet
        3. Dividi per std ImageNet
        
        Args:
            frame: Frame OpenCV (BGR) in range [0, 255]
        
        Returns:
            Frame normalizzato shape (3, H, W) per PyTorch
        """
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Normalizza a [0, 1]
        frame_rgb /= 255.0
        
        # Sottrai media ImageNet
        frame_rgb -= self.IMAGENET_MEAN
        
        # Dividi per std ImageNet
        frame_rgb /= self.IMAGENET_STD
        
        # Converti a formato PyTorch (C, H, W)
        frame_rgb = np.transpose(frame_rgb, (2, 0, 1))
        
        return frame_rgb
    
    def process_video(
        self,
        output_video_path: Optional[str] = None,
        return_tensors: bool = True,
    ) -> Union[np.ndarray, Any]:  # Any per torch.Tensor se disponibile
        """
        Processa il video e ritaglia tutti i frame.
        
        Args:
            output_video_path: Se specificato, salva il video ritagliato in questo percorso
            return_tensors: Se True, ritorna torch.Tensor se disponibile, altrimenti np.ndarray.
                           Se False, ritorna np.ndarray
        
        Returns:
            Tensor/Array di shape:
            - Se return_tensors=True e torch disponibile: (frames, 3, 200, 200) come torch.Tensor (normalizzato)
            - Altrimenti: (frames, 200, 200, 3) come np.ndarray [0, 255]
        
        Example:
            >>> cropper = VideoLandmarkCropper("video.mp4", landmarks)
            >>> cropped_tensor = cropper.process_video()  # (frames, 3, 224, 224)
            >>> # cropped_tensor è pronto per MobileNet
        """
        logger.info("Inizio elaborazione video...")
        
        # Denormalizza landmarks
        denormalized_landmarks = self._denormalize_landmarks()
        logger.info(f"Landmarks denormalizzati: range X=[{denormalized_landmarks[:, :, 0].min():.1f}, "
                   f"{denormalized_landmarks[:, :, 0].max():.1f}], "
                   f"range Y=[{denormalized_landmarks[:, :, 1].min():.1f}, "
                   f"{denormalized_landmarks[:, :, 1].max():.1f}]")
        
        # Calcola bounding box per frame
        x_min, x_max, y_min, y_max = self._compute_bounding_boxes_per_frame(
            denormalized_landmarks
        )
        
        # Applica padding e quadrato
        x_min, x_max, y_min, y_max = self._make_square_with_padding(
            x_min, x_max, y_min, y_max
        )
        
        # Applica smoothing temporale
        x_min, x_max, y_min, y_max = self._apply_temporal_smoothing(
            x_min, x_max, y_min, y_max
        )
        
        # Clip ai limiti del frame
        x_min, x_max, y_min, y_max = self._clip_to_frame_bounds(
            x_min, x_max, y_min, y_max
        )
        
        # Prepara output
        if return_tensors:
            output_array = np.zeros(
                (self.num_frames, 3, self.target_size[0], self.target_size[1]),
                dtype=np.float32
            )
        else:
            output_array = np.zeros(
                (self.num_frames, self.target_size[0], self.target_size[1], 3),
                dtype=np.uint8
            )
        
        # Prepara writer per video output (se richiesto)
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_video_path,
                fourcc,
                self.fps,
                self.target_size
            )
            if not out.isOpened():
                logger.warning(f"Impossibile creare video output: {output_video_path}")
                out = None
        else:
            out = None
        
        # Processa frame
        frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset a inizio video
        
        while frame_idx < self.num_frames:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Errore lettura frame {frame_idx}, interrotto.")
                break
            
            # Ritaglia il frame
            x_min_f, x_max_f = int(x_min[frame_idx]), int(x_max[frame_idx])
            y_min_f, y_max_f = int(y_min[frame_idx]), int(y_max[frame_idx])
            
            cropped_frame = frame[y_min_f:y_max_f, x_min_f:x_max_f]
            
            # Gestisci crop vuoto o troppo piccolo
            if cropped_frame.size == 0 or cropped_frame.shape[0] < 10 or cropped_frame.shape[1] < 10:
                logger.warning(f"Frame {frame_idx}: crop troppo piccolo, salto.")
                frame_idx += 1
                continue
            
            # Ridimensiona
            resized_frame = cv2.resize(cropped_frame, self.target_size)
            
            if return_tensors:
                # Normalizza per MobileNet
                normalized_frame = self._normalize_for_mobilenet(resized_frame)
                output_array[frame_idx] = normalized_frame
            else:
                # Converti a RGB e mantieni [0, 255]
                resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                output_array[frame_idx] = resized_frame_rgb
            
            # Scrivi nel video output se richiesto
            if out:
                out.write(resized_frame)
            
            if (frame_idx + 1) % 50 == 0:
                logger.info(f"Elaborati {frame_idx + 1}/{self.num_frames} frame")
            
            frame_idx += 1
        
        # Rilascia risorse
        self.cap.release()
        if out:
            out.release()
            logger.info(f"Video ritagliato salvato: {output_video_path}")
        
        # Converti in tensor se richiesto
        if return_tensors:
            if HAS_TORCH:
                output_tensor = torch.from_numpy(output_array)
                logger.info(
                    f"Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}, "
                    f"range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]"
                )
                return output_tensor
            else:
                logger.warning(
                    "PyTorch non disponibile, sono disponibili solo Output array (np.ndarray)"
                )
                return output_array
        else:
            logger.info(
                f"Output array shape: {output_array.shape}, dtype: {output_array.dtype}, "
                f"range: [{output_array.min()}, {output_array.max()}]"
            )
            return output_array
    
    def __del__(self):
        """Cleanup resource"""
        if hasattr(self, 'cap'):
            self.cap.release()


# ============================================================================
# Funzione di supporto semplificata (interfaccia veloce)
# ============================================================================

def crop_video_with_landmarks(
    video_path: str,
    landmarks: np.ndarray,
    output_path: Optional[str] = None,
    padding: float = 0.15,
    target_size: Tuple[int, int] = (200, 200),
    smoothing: str = "fixed",
    return_format: str = "tensor",
) -> Union[np.ndarray, Any]:  # Any per torch.Tensor se disponibile
    """
    Funzione semplificata per ritagliare video con landmarks.
    
    Args:
        video_path: Percorso al video RGB
        landmarks: Array shape (frames, num_landmarks, 2) in coordinate pixel
        output_path: Percorso per salvare il video ritagliato (opzionale)
        padding: Percentuale padding (default: 0.15 = 15%)
        target_size: Dimensioni finali (default: (200, 200) per MobileNet)
        smoothing: "fixed" (bounding box massima fissa, default) | "global" (media) | "moving" (media mobile)
        return_format: "tensor" (torch, normalizzato) o "array" (numpy, [0,255])
    
    Returns:
        Tensor/Array ritagliato e preprocessato
    
    Example:
        >>> landmarks = np.load("landmarks.npy")  # (frames, 2108, 2)
        >>> cropped = crop_video_with_landmarks(
        ...     "video.mp4",
        ...     landmarks,
        ...     output_path="output.mp4"
        ... )
        >>> print(cropped.shape)  # (frames, 3, 224, 224)
    """
    cropper = VideoLandmarkCropper(
        video_path,
        landmarks,
        padding_percent=padding,
        target_size=target_size,
        smoothing_method=smoothing,
    )
    
    return cropper.process_video(
        output_video_path=output_path,
        return_tensors=(return_format == "tensor"),
    )


# ============================================================================
# Script principale per test
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ritaglia video RGB usando landmarks normalizzati"
    )
    parser.add_argument("video_path", help="Percorso video input")
    parser.add_argument("landmarks_path", help="Percorso landmarks .npy")
    parser.add_argument(
        "--output-video",
        help="Percorso video output (opzionale)"
    )
    parser.add_argument(
        "--output-tensor",
        help="Percorso per salvare tensor PyTorch output"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Padding percentage (default: 0.15)"
    )
    parser.add_argument(
        "--smoothing",
        choices=["global", "moving"],
        default="global",
        help="Metodo smoothing temporale"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target size per MobileNet (default: 224 224)"
    )
    
    args = parser.parse_args()
    
    # Carica landmarks
    logger.info(f"Caricamento landmarks da: {args.landmarks_path}")
    landmarks = np.load(args.landmarks_path)
    logger.info(f"Landmarks shape: {landmarks.shape}")
    
    # Esegui cropping
    cropped = crop_video_with_landmarks(
        args.video_path,
        landmarks,
        output_path=args.output_video,
        padding=args.padding,
        target_size=tuple(args.target_size),
        smoothing=args.smoothing,
        return_format="tensor",
    )
    
    # Salva tensor se richiesto
    if args.output_tensor:
        logger.info(f"Salvataggio tensor in: {args.output_tensor}")
        if HAS_TORCH and isinstance(cropped, type(torch.tensor([]))):
            torch.save(cropped, args.output_tensor)
            logger.info("Tensor salvato con successo!")
        else:
            # Salva come numpy
            logger.warning("PyTorch non disponibile, salvataggio come file NumPy (.npy)")
            output_tensor_path = args.output_tensor.replace('.pt', '.npy').replace('.pth', '.npy')
            np.save(output_tensor_path, cropped)
            logger.info(f"Array salvato in: {output_tensor_path}")
    
    logger.info(f"Output shape: {cropped.shape}")
