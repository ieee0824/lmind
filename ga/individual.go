package ga

import (
	"os/exec"

	"github.com/ieee0824/lmind/config"
)

// Individual はGA個体（パラメータセット＋実行環境）
type Individual struct {
	ID      string
	Params  config.Params
	Port    int
	DataDir string
	Process *exec.Cmd
	Fitness float64
}
